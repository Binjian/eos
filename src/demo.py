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


def writexslx(v, a, p):
    # print (vehicle)
    df = pd.DataFrame({"v": v, "a": a, "p": p})
    writer = pd.ExcelWriter("data2.xlsx")
    df.to_excel(writer, index=False)
    writer.save()


def main():
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
        "ego_vehicle_filter": "vehicle.lincoln*",  # filter for defining ego vehicle
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
        "desired_speed": 8,  # desired speed (m/s)
        "max_ego_spawn_times": 200,  # maximum times to spawn ego vehicle
        "display_route": True,  # whether to render the desired route
        "pixor_size": 64,  # size of the pixor labels
        "pixor": False,  # whether to output PIXOR observation
        "file_path": "../data/straight.xodr",
    }

    # Set gym-carla environment
    env = gym.make("carla-v0", params=params)
    obs = env.reset()

    # initialize start time and counter
    start = time.time()
    counter = 0

    # initialize arrays to store data
    velocity = np.array([], dtype=float)
    acceleration = np.array([], dtype=float)
    pedal = np.array([], dtype=float)
    df = pd.read_excel("../data/WLTC.xlsx", sheet_name="WLTC")
    f_real = np.array(df["F"])

    print("simulation starts")

    while counter < 1800:
        action = [2.0, 0.0]
        obs, r, done, info = env.step(action)

        if time.time() - start > 1:
            print("-----------------------------")
            # print current time
            sec = int(time.time() - start) + counter
            print("time =", str(sec), "s")
            # print velocity
            v = obs["state"][2]
            print("vel=", str(v * 3.6), "km/h")
            # print acceleration
            a = obs["state"][4]
            print("acc=", str(a), "m/s2")
            # print throttle percentage
            print("throttle=", str(env.thro_percent), "%")
            # store data
            velocity = np.append(velocity, v)
            acceleration = np.append(acceleration, a)
            pedal = np.append(pedal, env.thro_percent)
            # reset time and add on counter
            start = time.time()
            counter = counter + 1

        if env.mode == 1 and done:
            obs = env.reset()


if __name__ == "__main__":
    main()
