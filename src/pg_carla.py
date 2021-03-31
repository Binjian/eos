#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import gym
import gym_carla
import carla
import pyglet
import time
import matplotlib.pyplot as plt
import gc

from carla import ColorConverter as cc

import numpy as np
import array
import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
import pandas as pd

# from openpyxl import load_workbook
from pathlib import Path

from pg_carla_agent import *

from controller import controller2d


def writexslx(v, a, p):
    # print (vehicle)
    df = pd.DataFrame({"v": v, "a": a, "p": p})
    writer = pd.ExcelWriter("data.xlsx")
    df.to_excel(writer, index=False)
    writer.save()


def prep_waypoints(waypoints_np, interp_distance_res):
    # Because the waypoints are discrete and our controller performs better
    # with a continuous path, here we will send a subset of the waypoints
    # within some lookahead distance from the closest point to the vehicle.
    # Interpolating between each waypoint will provide a finer resolution
    # path and make it more "continuous". A simple linear interpolation
    # is used as a preliminary method to address this issue, though it is
    # better addressed with better interpolation methods (spline
    # interpolation, for example).
    # More appropriate interpolation methods will not be used here for the
    # sake of demonstration on what effects discrete paths can have on
    # the controller. It is made much more obvious with linear
    # interpolation, because in a way part of the path will be continuous
    # while the discontinuous parts (which happens at the waypoints) will
    # show just what sort of effects these points have on the controller.
    # Can you spot these during the simulation? If so, how can you further
    # reduce these effects?

    # Linear interpolation computations
    # Compute a list of distances between waypoints
    # from the last waypoint to the last waypoint
    wp_distance = []  # distance array
    for i in range(1, waypoints_np.shape[0]):
        wp_distance.append(
            np.sqrt(
                (waypoints_np[i, 0] - waypoints_np[i - 1, 0]) ** 2
                + (waypoints_np[i, 1] - waypoints_np[i - 1, 1]) ** 2
            )
        )
    wp_distance.append(0)  # last distance is 0 because it is the distance
    # from the last waypoint to the last waypoint

    # Linearly interpolate between waypoints and store in a list
    wp_interp = []  # interpolated values
    # (rows = waypoints, columns = [x, y, v])
    wp_interp_hash = []  # hash table which indexes waypoints_np
    # to the index of the waypoint in wp_interp
    interp_counter = 0  # counter for current interpolated point index
    for i in range(waypoints_np.shape[0] - 1):
        # Add original waypoint to interpolated waypoints list (and append
        # it to the hash table)
        wp_interp.append(list(waypoints_np[i]))
        wp_interp_hash.append(interp_counter)
        interp_counter += 1

        # Interpolate to the next waypoint. First compute the number of
        # points to interpolate based on the desired resolution and
        # incrementally add interpolated points until the next waypoint
        # is about to be reached.
        num_pts_to_interp = int(
            np.floor(wp_distance[i] / float(interp_distance_res)) - 1
        )
        if num_pts_to_interp >= 1:
            wp_vector = waypoints_np[i + 1] - waypoints_np[i]
            wp_vector = wp_vector / np.linalg.norm(wp_vector)
            for j in range(num_pts_to_interp):
                next_wp_vector = interp_distance_res * float(j + 1) * wp_vector
                wp_interp.append(list(waypoints_np[i] + next_wp_vector))
                interp_counter += 1
    # add last waypoint at the end
    wp_interp.append(list(waypoints_np[-1]))
    wp_interp_hash.append(interp_counter)
    interp_counter += 1

    return wp_distance, wp_interp, wp_interp_hash

###
# Controller update (this uses the controller2d.py implementation)X
###

# To reduce the amount of waypoints sent to the controller,
# provide a subset of waypoints that are within some
# lookahead distance from the closest point to the car. Provide
# a set of waypoints behind the car as well.

# Find closest waypoint index to car. First increment the index
# from the previous index until the new distance calculations
# are increasing. Apply the same rule decrementing the index.
# The final index should be the closest point (it is assumed that
# the car will always break out of instability points where there
# are two indices with the same minimum distance, as in the
# center of a circle)
def new_waypoints_update(
    waypoints_np,
    wp_distance,
    wp_interp,
    wp_interp_hash,
    closest_index,
    current_x,
    current_y,
    interp_lookahead_distance,
):
    closest_distance = np.linalg.norm(
        np.array(
            [
                waypoints_np[closest_index, 0] - current_x,
                waypoints_np[closest_index, 1] - current_y,
            ]
        )
    )
    new_distance = closest_distance
    new_index = closest_index
    while new_distance <= closest_distance:
        closest_distance = new_distance
        closest_index = new_index
        new_index += 1
        if new_index >= waypoints_np.shape[0]:  # End of path
            break
        new_distance = np.linalg.norm(
            np.array(
                [
                    waypoints_np[new_index, 0] - current_x,
                    waypoints_np[new_index, 1] - current_y,
                ]
            )
        )
    new_distance = closest_distance
    new_index = closest_index
    while new_distance <= closest_distance:
        closest_distance = new_distance
        closest_index = new_index
        new_index -= 1
        if new_index < 0:  # Beginning of path
            break
        new_distance = np.linalg.norm(
            np.array(
                [
                    waypoints_np[new_index, 0] - current_x,
                    waypoints_np[new_index, 1] - current_y,
                ]
            )
        )
    closest_distance = np.linalg.norm(
        np.array(
            [
                waypoints_np[closest_index, 0] - current_x,
                waypoints_np[closest_index, 1] - current_y,
            ]
        )
    )
    # Once the closest index is found, return the path that has 1
    # waypoint behind and X waypoints ahead, where X is the index
    # that has a lookahead distance specified by
    # INTERP_LOOKAHEAD_DISTANCE
    waypoint_subset_first_index = closest_index - 1
    if waypoint_subset_first_index < 0:
        waypoint_subset_first_index = 0

    waypoint_subset_last_index = closest_index
    total_distance_ahead = 0
    while total_distance_ahead < interp_lookahead_distance:
        total_distance_ahead += wp_distance[waypoint_subset_last_index]
        waypoint_subset_last_index += 1
        if waypoint_subset_last_index >= waypoints_np.shape[0]:
            waypoint_subset_last_index = waypoints_np.shape[0] - 1
            break

    # Use the first and last waypoint subset indices into the hash
    # table to obtain the first and last indicies for the interpolated
    # list. Update the interpolated waypoints to the controller
    # for the next controller update.
    new_waypoints = wp_interp[
        wp_interp_hash[waypoint_subset_first_index]: wp_interp_hash[
            waypoint_subset_last_index
        ]
        + 1
    ]

    return new_waypoints, closest_index


def main():
    # parameters for the gym_carla environment
    params = {
        "number_of_vehicles": 0,
        "number_of_walkers": 0,
        "display_size": 256,  # screen size of bird-eye render
        "max_past_step": 1,  # the number of past steps to draw
        "dt": 0.1,  # time interval between two frames
        "discrete": False,  # whether to use discrete control space
        "discrete_throttle": [-3.0, 0.0, 3.0],  # discrete value of accelerations
        "discrete_steer": [-0.2, 0.0, 0.2],  # discrete value of steering angles
        "continuous_accel_range": [-3.0, 3.0],  # continuous acceleration range
        "continuous_steer_range": [-0.3, 0.3],  # continuous steering angle range
        # "ego_vehicle_filter": "vehicle.tesla.model3",  # filter for defining ego vehicle
        # "ego_vehicle_filter": "vehicle.carlamotors.carlacola",  # filter for defining ego vehicle
        # "ego_vehicle_filter": "vehicle.lincoln*",  # filter for defining ego vehicle
        "ego_vehicle_filter": "vehicle.mustang.mustang",  # filter for defining ego vehicle
        "carlaserver": "192.168.60.80",  # carla server ip address
        "port": 2000,  # connection port
        "town": "Town03",  # which town to simulate
        "task_mode": "random",  # mode of the task, [random, roundabout (only for Town03)]
        "max_time_episode": 1000,  # maximum timesteps per episode
        "max_waypt": 12,  # maximum number of waypoints
        "obs_range": 32,  # observation range (meter)
        "lidar_bin": 0.125,  # bin size of lidar sensor (meter)
        "d_behind": 12,  # distance behind the ego vehicle (meter)
        "out_lane_thres": 2.0,  # threshold for out of lane
        "desired_speed": 40,  # desired speed (m/s)
        "max_ego_spawn_times": 200,  # maximum times to spawn ego vehicle
        "display_route": True,  # whether to render the desired route
        "pixor_size": 64,  # size of the pixor labels
        "pixor": False,  # whether to output PIXOR observation
        # "odrfile_path": "../data/highring.xodr",  # opendrive map file to be loaded
        "odrfile_path": "../data/racetrack.xodr",  # opendrive map file to be loaded
        "mdlfile_path": "../data/pg-carla-save.p",  # pg model file path
        # "wpfilehuman_path": "../data/waypoint_highring.xls",  # waypoint excel file
        "wpfilehuman_path": "../data/waypoint_racetrack.xls",  # waypoint excel file
        # "wpfilestd_path": "../data/waypoint_racetrack.xls",  # waypoint excel file
        "wpfilesheet": "set",  # waypoint excel file
        #  simulation configurations
        "iter_for_sim_timestep": 10,  # no. iterations to compute approx sim timestep
        "wait_time_before_start": 5.00,  # game seconds (time before controller start)
        "total_run_time": 200.00,  # game seconds (total runtime before sim end)
        "total_frame_buffer": 300,  # number of frames to buffer after total runtime
        "num_pedestrians": 0,  # total number of pedestrians to spawn
        "num_vehices": 0,  # total number of vehicles to spawn
        "seed_pedestrians": 0,  # seed for pedestrian spawn randomizer
        "seed_vehicles": 0,  # seed for vehicle spawn  randomizer
        "player_start_index": 1,  # spawn index for player (keep to 1)
        "figsize_x_inches": 8,  # x figure size of feedback in inches
        "figsize_y_inches": 8,  # y figure size of feedback in inches
        "plot_left": 0.1,  # in fractions of figure width and height
        "plot_bot": 0.1,
        "plot_width": 0.8,
        "plot_height": 0.8,
        "waypoints_filename": "racetrack_waypoints.txt",  # waypoint file to load
        "dist_threshold_to_last_waypoint": 2.0,  # some distance from last position before simulation ends path interpolation parameters
        "interp_max_points_plot": 10,  # number of points used for displaying  lookahead path
        "interp_lookahead_distance": 20,  # lookahead in meters
        "interp_distance_res": 0.01,  # distance between interpolated points
        # controller output directory
        "controller_output_folder": "../output/",
    }

    #############################################
    # Load Waypoints
    #############################################
    # Opens the waypoint file and stores it to "waypoints"
    excel_data_df = pd.read_excel(
        params["wpfilehuman_path"], sheet_name=params["wpfilesheet"]
    )
    waypoints_human = excel_data_df.to_numpy()
#    waypoints_human[:, 2] = (
#        waypoints_human[:, 2]
#    )  ## seems the human behavior should be scaled by 4 in velocity
#    excel_data_df = pd.read_excel(
#        params["wpfilestd_path"], sheet_name=params["wpfilesheet"]
#    )
#     waypoints_std = excel_data_df.to_numpy()
    waypoints_np = waypoints_human[30:, :3]
    waypoints_list = waypoints_np.tolist()
    #  preprocesses waypoints, interplolation
    [wp_distance, wp_interp, wp_interp_hash] = prep_waypoints(
        waypoints_np, params["interp_distance_res"]
    )
    total_episode_frames = waypoints_np.shape[0]
#    total_episode_frames = int(
#        waypoints_np.shape[0] / params["dt"]
#    )  # standard data has only 900 data points
    # total_episode_frames = int(total_episode_frames/10)  # only 90 seconds

    # Set gym-carla environment
    env = gym.make("carla-v0", params=params)

    # Simulation initialization, episode running reward
    print("simulation starts")
    acts = []
    running_reward = None
    reward_sum = 0
    episode_number = 0
    counter = 0
    episode_reward = []

    ############################################
    # Controller 2D Class Declaration
    ############################################
    # This is where we take the controller2d.py class
    # and apply it to the simulator
    controller = controller2d.Controller2D(waypoints_list)
    try:
        while True:
            # episode initialization
            xs, hs, dlogps, drs = [], [], [], []
            pos_x, pos_y, yaw_z, velocity, acceleration, timing = [], [], [], [], [], []
            epi_kp, epi_ki, epi_kd = [], [], []

            obs = env.reset()

            current_yaw = obs["state"][1]
            current_speed = obs["state"][2]
            current_a = obs["state"][4]
            current_x = obs["state"][5]
            current_y = obs["state"][6]
            x = [current_speed, current_a]
            # prev_x = None  #  reserved for computing the difference frame
            velocity.append(current_speed)
            acceleration.append(current_a)
            timing.append(0)
            pos_x.append(current_x)
            pos_y.append(current_y)
            yaw_z.append(current_yaw)

            kp, ki, kd = controller.get_pid()
            epi_kp.append(kp)
            epi_ki.append(ki)
            epi_kd.append(kd)

            # Iterate the frames until the end of the waypoints is reached or
            # the TOTAL_EPISODE_FRAMES is reached. The controller simulation then
            # ouptuts the results to the controller output directory.
            closest_index = 0  # Index of waypoint that is currently closest to
            # the car (assumed to be the first index)
            closest_distance = 0  # Closest distance of closest waypoint to car

            cmd_throttle, cmd_steer, cmd_brake = controller.get_commands()
            action = [cmd_throttle, cmd_steer, cmd_brake]
            acts.append(action)  # logging the actions taken for visualization
            # halting_counter = 0
            skip_first_frame = True

            reached_the_end = False
            for frame in range(total_episode_frames):  # frame is the timestamps since we need to handle halting waypoints.
                # TODO test multiple frames of observation to get the optimal length of frames for observation
                # aprob, h = policy_forward(x)
                # # record various intermediates (needed later for backprop)
                # xs.append(x)  # observation
                # hs.append(h)  # hidden state
                # # action = [2.0, 0.0]
                # action_index = np.random.choice(A3, 1, aprob.tolist())[0]

                # kp_ind = int(action_index / A2)
                # ki_ind = int((action_index % A2) / A)
                # kd_ind = action_index % A

                # kp = KP_space[kp_ind]
                # ki = KI_space[ki_ind]
                # kd = KD_space[kd_ind]
                # controller.set_pid(kp, ki, kd)
                # controller.set_pid(1, 1, 0.1)

                # closest_index = frame
                new_waypoints, closest_index = new_waypoints_update(
                    waypoints_np,
                    wp_distance,
                    wp_interp,
                    wp_interp_hash,
                    closest_index,
                    current_x,
                    current_y,
                    params["interp_lookahead_distance"],
                )
                controller.update_waypoints(new_waypoints)

                # # handling halting waypoints!
                # if waypoints_np[closest_index, 2] < 1e-2:
                #     halting_counter += 1
                #
                # if halting_counter == 10:  # after 10*dt, ignore one halting waypoints
                #     halting_counter = 0
                #     closest_index += 1

                # Update the other controller values and controls
                current_timestamp = frame * params["dt"]
                controller.update_values(
                    current_x,
                    current_y,
                    current_yaw,
                    current_speed,
                    current_timestamp,
                    frame
                )
                controller.update_controls()
                cmd_throttle, cmd_steer, cmd_brake = controller.get_commands()
                # cmd_throttle = 0.99
                action = [cmd_throttle, cmd_steer, cmd_brake]

                acts.append(action)  # logging the actions taken for visualization

                # # y = 1 if action == 2 else 0 # a "fake label"
                # y = np.zeros(A ** 3)
                # y[action_index] = 1
                # # grad that encourages the action that was taken to be taken
                # dlogps.append(y - aprob)
                # # (see http://cs231n.github.io/neural-networks-2/#losses if confused)

                if skip_first_frame and frame == 0:
                    pass

                # step the environment and get new measurements
                obs, r, done, info = env.step(action)

                # print("-----------------------------")
                current_yaw = obs["state"][1]
                current_speed = obs["state"][2]
                current_a = obs["state"][4]
                current_x = obs["state"][5]
                current_y = obs["state"][6]
                x = [current_speed, current_a]
                if frame % 10 == 0:
                    # print("closest Index:%d, desired Speed:%f" % (closest_index, waypoints_np[closest_index, 2]))
                    print("closest Index:%d" % (closest_index))
                    print("time:%f, x=%f, y=%f, v=%f, a=%f " % (frame/10, current_x, current_y, current_speed, current_a))
                    print("throttle:%f,steer:%f,brake:%f" % (cmd_throttle, cmd_steer, cmd_brake))

                # prev_x = None  #  reserved for computing the difference frame
                velocity.append(current_speed)
                acceleration.append(current_a)
                timing.append(current_timestamp)
                pos_x.append(current_x)
                pos_y.append(current_y)
                yaw_z.append(current_yaw)
                epi_kp.append(kp)
                epi_ki.append(ki)
                epi_kd.append(kd)
                r_engy_consump = -(current_a ** 2)
                r_time_lapse = -1  # for fixed trip length consider + r_time_lapse
                # r_trip_length = wp_distance[frame]  # for fixed time range consider + r_trip_length
                reward = r_engy_consump  # + r_time_lapse
                reward_sum += reward

                # record reward (has to be done after we call step() to get reward for
                # previous action)
                drs.append(reward)

                # Find if reached the end of waypoint. If the car is within
                # DIST_THRESHOLD_TO_LAST_WAYPOINT to the last waypoint,
                # the simulation will end.
                dist_to_last_waypoint = np.linalg.norm(
                    np.array(
                        [
                            waypoints_np[-1][0] - current_x,
                            waypoints_np[-1][1] - current_y,
                        ]
                    )
                )
                if dist_to_last_waypoint < params["dist_threshold_to_last_waypoint"]:
                    reached_the_end = True
                if reached_the_end:
                    break
                if r_time_lapse > total_episode_frames * 1.2:
                    break

            ###### Episode done ######
            # if not reaching the end, give a large loss
            if not reached_the_end:
                reward = -1000
                drs.append(reward)
            episode_number = episode_number + 1
            episode_reward.append(reward_sum)

            # episode velocity comparison between real, wltc standard, human operator,
            plt.subplot(231)
            plt.plot(
                timing,
                velocity,
                "r-",
                timing,
                waypoints_np[:, 2],
                "g--",
            )
            plt.show(block=False)
            # epdisode actions:throttle, steer and brake
            plt.subplot(232)
            acts_np = np.array(acts)
            plt.plot(
                timing,
                acts_np[:, 0],
                "r-",
                timing,
                acts_np[:, 1],
                "g-",
                acts_np[:, 2],
                "b-",
            )
            plt.show(block=False)
            # episode pid
            plt.subplot(235)
            plt.plot(
                timing, epi_kp, "r-", timing, epi_ki, "g-", epi_kd, "b-",
            )
            plt.plot(episode_reward, "r*")
            plt.show(block=False)
            # epdisode accelerate
            plt.subplot(233)
            plt.plot(acceleration, "r-")
            plt.show(block=False)
            # rewards for this episode
            plt.subplot(234)
            plt.plot(timing, drs, "r-")
            plt.show(block=False)
            # episode rewards
            plt.subplot(236)
            plt.plot(episode_reward, "r*")
            plt.show(block=False)

            # stack together all inputs, hidden states, action gradients, and
            epx = np.vstack(xs)
            eph = np.vstack(hs)
            epdlogp = np.vstack(dlogps)
            epr = np.vstack(drs)

            # compute the discounted reward backwards through time
            discounted_epr = discount_rewards(epr)
            # standardize the rewards to be unit normal (helps control the gradient
            # estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            # modulate the gradient with advantage (PG magic happens right here.)
            epdlogp *= discounted_epr
            grad = policy_backward(epx, eph, epdlogp)
            for k in model:
                grad_buffer[k] += grad[k]  # accumulate grad over batch

            if episode_number % batch_size == 0:
                for k, v in list(model.items()):
                    g = grad_buffer[k]  # gradient
                    rmsprop_cache[k] = (
                        decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                    )
                    model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                    # reset batch gradient buffer
                    grad_buffer[k] = np.zeros_like(v)

            # boring book-keeping
            running_reward = (
                reward_sum
                if running_reward is None
                else running_reward * 0.99 + reward_sum * 0.01
            )
            print(
                "resetting env. episode %d reward total was %f. running mean: %f"
                % (episode_number, reward_sum, running_reward)
            )

            reward_sum = 0
            if episode_number % 100 == 0:
                pickle.dump(model, open("data/pg-carla-save.p", "wb"))

            # intialize the log arrays
            xs, hs, dlogps, drs = [], [], [], []  # reset array memory
            acts = []  # could be plotted to compare with human throttle input
            velocity, acceleration, timing = [], [], []
            xs, hs, dlogps, drs = [], [], [], []
            pos_x, pos_y, yaw_z, velocity, acceleration, timing = [], [], [], [], [], []
            epi_kp, epi_ki, epi_kd = [], [], []

            # reset episode
            obs = env.reset()  # reset env
            start = time.time()
            epi_start = start
            vel = obs["state"][2]
            acc = obs["state"][4]
            x = [vel, acc]
            velocity.append(vel)
            acceleration.append(acc)
            timing.append(0)
            # prev_x = None  #  reserved for computing the difference frame
    except:  # FIXME test error capture
        env.deleteElement()
        gc.collect()
    else:
        print("Simulation ended!")
        gc.collect()


if __name__ == "__main__":
    main()
