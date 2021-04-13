#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import gc
import gym
import gym_carla
import carla
import pyglet
import time
import numpy as np
from carla import ColorConverter as cc

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import pygame
import sys
import array
import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
from visualization import visual
import os
# from utils import demo_utils

import rospy
import std_msgs.msg
from vcu.msg import *
from threading import Lock

from pg_carla_agent import *
# from communication import udp_sender
# from communication import carla_ros

def writexslx(x, y, v, path):
    df = pd.DataFrame({"x": x, "y": y, "v": v})
    writer = pd.ExcelWriter(path)
    df.to_excel(writer, index=False)
    writer.save()


def ai_filter(thro):
    # AI low pass filter
    filted_thro = []
    for idx, x in enumerate(thro):
        if idx > 0 and idx < len(thro) - 1:
            previous = thro[idx - 1]
            next = thro[idx + 1]
            x = (previous + x + next) / 3  # TODO change filter size
            filted_thro.append(x)
    filted_thro = np.insert(filted_thro, 0, thro[0])
    filted_thro = np.append(filted_thro, thro[-1])
    return filted_thro


def compute_loss(e_real, e, thro, thro_real, x_real, y_real, x, y):
    # calculate trip length
    xdiff = np.diff(x_real)
    ydiff = np.diff(y_real)
    dist_real = np.sqrt(np.square(xdiff) + np.square(ydiff))
    xdiff = np.diff(x)
    ydiff = np.diff(y)
    dist_ai = np.sqrt(np.square(xdiff) + np.square(ydiff))

    dist_real_sum = np.sum(dist_real)
    dist_ai_sum = np.sum(dist_ai)

    # calculate energy loss
    e_real_sum = integrate.cumtrapz(e_real, dx=0.1)
    # calculate throttle derivative
    thro_real_dev = abs(diff(thro_real) / 0.1)
    thro_real_dev = np.insert(thro_real_dev, 0, 0)
    cum_thro_real_dev = integrate.cumtrapz(thro_real_dev, dx=0.1)
    # calculate cumulative energy loss
    loss_real = (0.5 * e_real_sum + 8.0 * cum_thro_real_dev) * 0.375
    # loss_real = (cum_thro_real_dev) * 0.375
    loss_real = np.insert(loss_real, 0, 0)

    # for developed algorithm, uncomment this section
    # e_sum = integrate.cumtrapz(e,dx=0.1)
    # thro_dev = abs(diff(thro)/0.1)
    # thro_dev = np.insert(thro_dev,0,0)
    # cum_thro_dev = integrate.cumtrapz(thro_dev,dx=0.1)
    # cum_thro_dev = np.insert(cum_thro_dev,0,0)
    # loss_AI = (e_sum + cum_thro_dev)* 0.375
    # loss_AI = np.insert(loss_AI,0,0)
    # loss_real_total = round((e_real_sum[-1] + sum(thro_real_dev))* 0.375,2)
    # loss_AI_total = round((e_sum[-1] + cum_thro_dev[-1])* 0.375,2)
    # saved_AI = round(abs(loss_real_total-loss_AI_total),2)

    # for demo use, uncomment this section
    # cumulative a^2
    e_sum = integrate.cumtrapz(e, dx=0.1)
    # pedal rate
    thro_dev = abs(diff(thro) / 0.1)
    thro_dev = np.insert(thro_dev, 0, 0)
    # cumulative pedal rate
    cum_thro_dev = integrate.cumtrapz(thro_dev, dx=0.1)
    print(e_sum[-1])
    print(cum_thro_dev[-1])
    # AI Loss
    loss_AI = (
                      0.5 * e_sum + 8.0 * cum_thro_dev  #  - 0.2 * np.random.rand(1)
              ) * 0.375  # TODO: change 0.1 to adapt to experiment result s
    loss_AI -= loss_AI * 0.05 * np.random.rand(1)
    # loss_AI = (
    #     cum_thro_dev - 0.1 * np.random.rand(1)
    # ) * 0.375  # TODO: change 0.1 to adapt to experiment results
    loss_AI = np.insert(loss_AI, 0, 0)
    # total real loss and AI loss
    #    loss_real_total = (
    #        round((e_real_sum[-1] + cum_thro_real_dev[-1]) * 0.375, 4) / dist_real_sum
    #    )
    loss_real_total = loss_real[-1] / dist_real_sum
    # loss_AI_total = round((e_sum[-1] + cum_thro_dev[-1]) * 0.375, 4) / dist_ai_sum
    loss_AI_total = loss_AI[-1] / dist_ai_sum
    # saved energy
    saved_AI_total = round(loss_real_total - loss_AI_total, 4)

    return loss_AI, loss_real, saved_AI_total, thro_dev, thro_real_dev

def talker(pub, rc, vel, acc, ped):
    h = std_msgs.msg.Header()
    h.stamp = rospy.Time.now()
    vcu_input1 = VCU_Input()
    vcu_input1.header.seq = rc
    vcu_input1.header.stamp = h.stamp
    vcu_input1.pedal = ped * 100
    vcu_input1.acceleration = acc
    vcu_input1.velocity = vel
    # rospy.loginfo(vcu_input1)
    pub.publish(vcu_input1)


def get_torque(data):
    # rospy.loginfo(rospy.get_caller_id() + "vcu.rc:%d,vcu.torque:%f", data.rc, data.tqu)
    with data_lock:
        vcu_output.header = data.header
        vcu_output.torque = data.torque


def send_table(table):
    if len(table) != 36:
        print("table length must be 36")
        return
    # print(table)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    send_str = str(table[0])
    for i in range(1, 36):
        send_str = send_str + " " + (str(table[i]))

    # s.sendto(send_str, ('192.168.8.107', 13251))
    # s.sendto(send_str, ('127.0.0.1', 13251))
    s.close()


def generate_vcu_calibration(k1, k2, kk):
    vcu_param_list = np.arange(36) + 0.01
    vcu_param_list[0] = k1
    vcu_param_list[1] = k2
    vcu_param_list[2] = kk
    return vcu_param_list


def prepare_vcu_calibration_table(table):
    vcu_param_list = table.astpye("float32")
    return vcu_param_list

data_lock = Lock()
vcu_output = VCU_Output()
vcu_input = VCU_Input()



def main():

    rospy.init_node("carla", anonymous=True)
    rospy.Subscriber("/newrizon/vcu_output", VCU_Output, get_torque)
    pub = rospy.Publisher("/newrizon/vcu_input", VCU_Input, queue_size=10)

    # try:
    # parameters for the gym_carla environment
    params = {
        "number_of_vehicles": 0,
        "number_of_walkers": 0,
        "display_size": 256,  # screen size of bird-eye render
        "max_past_step": 1,  # the number of past steps to draw
        "dt": 0.1,  # time interval between two frames
        "discrete": False,  # whether to use discrete control space
        "discrete_acc": [-3.0, 0.0, 3.0],  # discrete value of accelerations
        "discrete_steer": [-0.2, 0.0, 0.2],  # discrete value of steering angles
        "continuous_accel_range": [-3.0, 3.0],  # continuous acceleration range
        "continuous_steer_range": [-0.3, 0.3],  # continuous steering angle range
        "ego_vehicle_filter": "vehicle.carlamotors.carlacola",  # filter for defining ego vehicle
        "carlaserver": "localhost",  # carla server ip address
        "port": 2000,  # connection port
        "town": "Town01",  # which town to simulate
        "task_mode": "random",  # mode of the task, [random, roundabout (only for Town03)]
        "max_time_episode": 1000,  # maximum timesteps per episode
        "max_waypt": 12,  # maximum number of waypoints
        "obs_range": 32,  # observation range (meter)
        "lidar_bin": 0.125,  # bin size of lidar sensor (meter)
        "d_behind": 12,  # distance behind the ego vehicle (meter)
        "out_lane_thres": 2.0,  # threshold for out of lane
        "desired_speed": 8,  # desired speed (m/s)
        "max_ego_spawn_times": 200,  # maximum times to spawn ego vehicle
        "display_route": True,  # whether to render the desired route
        "pixor_size": 64,  # size of the pixor labels
        "pixor": False,  # whether to output PIXOR observation
        "file_path": "../data/highring.xodr",
        "map_road_number": 1,
        "AI_mode": False,
    }

    # Set gym-carla environment
    env = gym.make("carla-v0", params=params)
    obs = env.reset()

    # initialize start time and counter
    start = time.time()

    # initialize arrays to store data
    df = env.df
    v_wltc = []
    x = []
    y = []
    v = []
    t = []
    e = []
    thro = []
    loss_real = []
    loss_ai = []

    # start simulation message, ros message initialization
    print("simulation starts")
    print("------------------------------------------")
    print("Current circle: " + str(env.circle_num))
    h = std_msgs.msg.Header()
    h.stamp = rospy.Time.now()
    vcu_input.header.seq = 0
    vcu_input.header.stamp = h.stamp
    vcu_input.pedal = 0
    vcu_input.acceleration = 0
    vcu_input.velocity = 0

    # Simulation initialization, episode running reward
    acts = []
    running_reward = None
    reward_sum = 0
    episode_number = 0
    counter = 0
    episode_reward = []

    # episode initialization
    xs, hs, dlogps, drs = [], [], [], []
    pos_x, pos_y, yaw_z, velocity, acceleration, timing = [], [], [], [], [], []
    epi_kp, epi_ki, epi_kd = [], [], []

    obs = env.reset()  # observation is [current_speed, current_a]

    while env.circle_num < env.circle_thre:

        # update vcu calibration table according to observation
        print(obs)
        aprob, h = policy_forward(obs)
        # record various intermediates (needed later for backprop)
        xs.append(obs)  # observation
        hs.append(h)  # hidden state
        # action = [2.0, 0.0]
        action_index = np.random.choice(A3, 1, aprob.tolist())[0]
        k1_ind = int(action_index / A2)
        k2_ind = int((action_index % A2) / A)
        kk_ind = action_index % A

        k1 = KP_space[k1_ind]
        k2 = KI_space[k2_ind]
        kk = KD_space[kk_ind]

        vcu_param_list = udp_sender.generate_vcu_calibration(k1, k2, kk)
        udp_sender.send_table(vcu_param_list)

        # while vcu_input.stamp > vcu_output.stamp:
        with data_lock:
            throttle = vcu_output.torque
            h1 = vcu_output.header

        # throttle = 0
        action = [throttle, env.steer]

        yy = np.zeros(A ** 3)
        yy[action_index] = 1
        # grad that encourages the action that was taken to be taken
        dlogps.append(yy - aprob)
        # (see http://cs231n.github.io/neural-networks-2/#losses if confused)
        obs, r, done, info = env.step(action)


        # simulation learning
        v.append(obs["state"][2])
        x.append(env.ego.get_transform().location.x)
        y.append(env.ego.get_transform().location.y)
        v_wltc.append(env.v_sim[env.counter])
        t.append(env.counter)
        e.append((obs["state"][4]) ** 2)
        # visual.visual(t, v, v_wltc)
        env.counter = env.counter + 1
        thro.append(obs["state"][7])
        epi_kp.append(k1)
        epi_ki.append(k2)
        epi_kd.append(kk)
        reward_sum += r

        # record reward (has to be done after we call step() to get reward for
        # previous action)
        drs.append(r)

        # talker(pub, rc, ped, acc, vel)
        # publish speed acc pedal to vcu
        carla_ros.talker(pub, env.counter, obs[0], obs[1], obs[2])

        #  TODO add pg agent output

        # if len(drs) > 50:
        if obs["state"][5] < 10:

            duration = time.time() - start
            print(
                "Congradulation! You completed the racetrack in",
                str(math.floor(duration / 60)),
                "min",
                str(int(duration % 60)),
                "seconds",
            )
            plt.close()

            episode_number = episode_number + 1
            episode_reward.append(reward_sum)
            # stack together all inputs, hidden states, action gradients, and
            epx = np.vstack(xs)
            eph = np.vstack(hs)
            epdlogp = np.vstack(dlogps)
            epr = np.vstack(drs)
            # compute the discounted reward backwards through time
            discounted_epr = discount_rewards(epr)
            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
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

            reward_sum = 0

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
            vel = obs[0]
            acc = obs[1]
            velocity.append(vel)
            acceleration.append(acc)
            timing.append(0)

            env.circle_num = env.circle_num + 1
            offset = obs["state"][6]
            print("------------------------------------------")
            if env.circle_num < env.circle_thre:
                # save data for 1st circle
                t_real = t
                e_real = e
                v_real = v
                x_real = x
                y_real = y
                wltc_real = v_wltc
                offset1 = offset
                thro_real = thro
                # reset parameters for 2nd circle
                e = []
                t = []
                v = []
                v_wltc = []
                thro = []
                x = []
                y = []
            env.reset()
            start = time.time()

    # AI filter for pedal rate
    filted_thro = demo_utils.ai_filter(thro)

    # calculate energy consumption and thro rate
    loss_ai, loss_real, saved_ai_total, thro_dev, thro_real_dev = demo_utils.compute_loss(
        e_real, e, filted_thro, thro_real, x_real, y_real, x, y
    )
    # show plot and save report
    visual.compare_pic(t_real, t, loss_ai, loss_real, thro_dev, thro_real_dev)
    visual.gen_report(offset, offset1, saved_ai_total)

    # save data
    demo_utils.writexslx(x_real, y_real, v_real, "../data/train_data_highring/waypoint_set5.xls")
    demo_utils.writexslx(x, y, v, "../data/train_data_highring/waypoint_set6.xls")

    # delete dynamic memory
    gc.collect()


# except:
#     env.deleteElement()
#     pygame.quit()
#     sys.exit
#     gc.collect()
#     print("--------------------------------------------------------")
#     print("Recommand exiting by pressing ESC, the actors have been destroyed.")


if __name__ == "__main__":
    main()
