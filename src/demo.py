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
from visualization.visual import visual, compare_pic, gen_report
import os
from utils.demo_utils import writexslx, ai_filter, compute_loss

import rospy
import std_msgs.msg
from comm.vcu.msg import *
from threading import Lock

from pg_carla_agent import *
from comm.udp_sender import (
    send_table,
    prepare_vcu_calibration_table,
)
from comm.vcu_calib_generator import (
    generate_vcu_calibration,
    generate_lookup_table,
)

from comm.carla_ros import talker
import socket

data_lock = Lock()
vcu_output = VCU_Output()
vcu_input = VCU_Input()

def get_torque(data):
    # rospy.loginfo(rospy.get_caller_id() + "vcu.rc:%d,vcu.torque:%f", data.rc, data.tqu)
    with data_lock:
        vcu_output.header = data.header
        vcu_output.torque = data.torque

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

    vcu_calib_table_row = 17  # number of pedal steps
    vcu_calib_table_col = 21  # numnber of velocity steps
    vcu_calib_table_size = vcu_calib_table_row * vcu_calib_table_col
    pedal_range = [0, 1.0]
    velocity_range = [0, 20.0]

    vcu_calib_table = generate_vcu_calibration(
        vcu_calib_table_row, pedal_range, vcu_calib_table_col, velocity_range
    )
    vcu_lookup_table = generate_lookup_table(
        pedal_range, velocity_range, vcu_calib_table
    )

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
    throt = obs[2]
    obs = obs[0:2]
    rc = 0
    vel = obs[0]
    acc = obs[1]

    while env.circle_num < env.circle_thre:

        # update vcu calibration table according to observation
        # print(obs)
        throttle = vcu_lookup_table(
            throt, vel
        )  # look up vcu table with pedal and speed  for throttle request

        # send_table(vcu_calib_table)
        # while vcu_input.stamp > vcu_output.stamp:
        with data_lock:
            throttle = vcu_output.torque
            h1 = vcu_output.header

        # throttle = throt
        action = [throttle, 0]
        obs, r, done, info = env.step(action)
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
        vcu_calib_table[0] = k1
        vcu_calib_table[1] = k2
        vcu_calib_table[2] = kk

        vcu_lookup_table = generate_lookup_table(
            pedal_range, velocity_range, vcu_calib_table
        )

        yy = np.zeros(A ** 3)
        yy[action_index] = 1
        # grad that encourages the action that was taken to be taken
        dlogps.append(yy - aprob)
        # (see http://cs231n.github.io/neural-networks-2/#losses if confused)
        throt = obs[2]
        vel = obs[0]
        acc = obs[1]
        obs = obs[0:2]

        # simulation learning
        v.append(obs[0])
        x.append(env.ego.get_transform().location.x)
        y.append(env.ego.get_transform().location.y)
        v_wltc.append(env.v_sim[env.counter])
        t.append(env.counter)
        e.append(obs[1] ** 2)
        visual(t, v, v_wltc)
        env.counter = env.counter + 1
        thro.append(throt)
        epi_kp.append(k1)
        epi_ki.append(k2)
        epi_kd.append(kk)
        reward_sum += r

        # record reward (has to be done after we call step() to get reward for
        # previous action)
        drs.append(r)

        talker(pub, rc, throt, acc, vel)
        rc += 1
        # publish speed acc pedal to vcu
        # carla_ros.talker(pub, env.counter, obs[0], obs[1], throt)

        #  TODO add pg agent output

        # if len(drs) > 50:
        if done:

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
            offset = obs["state"][5]
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
                rc = 0
            env.reset()
            start = time.time()

    # AI filter for pedal rate
    filted_thro = ai_filter(thro)

    # calculate energy consumption and thro rate
    loss_ai, loss_real, saved_ai_total, thro_dev, thro_real_dev = compute_loss(
        e_real, e, filted_thro, thro_real, x_real, y_real, x, y
    )
    # show plot and save report
    compare_pic(t_real, t, loss_ai, loss_real, thro_dev, thro_real_dev)
    gen_report(offset, offset1, saved_ai_total)

    # save data
    writexslx(x_real, y_real, v_real, "../data/train_data_highring/waypoint_set5.xls")
    writexslx(x, y, v, "../data/train_data_highring/waypoint_set6.xls")

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
