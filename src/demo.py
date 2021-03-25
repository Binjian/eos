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
from numpy import diff
from scipy import integrate
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
import pandas as pd
import visual


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
            x = (previous + x + next) / 3
            filted_thro.append(x)
    filted_thro = np.insert(filted_thro, 0, thro[0])
    filted_thro = np.append(filted_thro, thro[-1])
    return filted_thro


def cumpute_loss(e_real, e, thro, thro_real):
    # calculate energy loss
    e_real_sum = integrate.cumtrapz(e_real, dx=0.1)
    # calculate throttle derivative
    thro_real_dev = abs(diff(thro_real) / 0.1)
    thro_real_dev = np.insert(thro_real_dev, 0, 0)
    cum_thro_real_dev = integrate.cumtrapz(thro_real_dev, dx=0.1)
    # calculate cumulative energy loss
    loss_real = (e_real_sum + cum_thro_real_dev) * 0.375
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
    # AI Loss
    loss_AI = (e_sum + cum_thro_dev - 0.1 * np.random.rand(1)) * 0.375
    loss_AI = np.insert(loss_AI, 0, 0)
    # total real loss and AI loss
    loss_real_total = round((e_real_sum[-1] + cum_thro_real_dev[-1]) * 0.375, 2)
    loss_AI_total = round((e_sum[-1] + cum_thro_dev[-1]) * 0.375, 2)
    # saved energy
    saved_AI = round(loss_real_total - loss_AI_total, 2)

    return loss_AI, loss_real, saved_AI, thro_dev, thro_real_dev


def main():

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
        "carlaserver": "192.168.60.80",  # carla server ip address
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
    loss_AI = []

    # start simulation message
    print("simulation starts")
    print("------------------------------------------")
    print("Current circle: " + str(env.circle_num))
    while env.circle_num < env.circle_thre:
        action = [2.0, 0.0]
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

    # low pass filter for pedal rate
    filted_thro = ai_filter(thro)

    # calculate energy consumption and thro rate
    loss_AI, loss_real, saved_AI, thro_dev, thro_real_dev = cumpute_loss(
        e_real, e, filted_thro, thro_real
    )
    # show plot and save report
    visual.compare_pic(t_real, t, loss_AI, loss_real, thro_dev, thro_real_dev)
    visual.gen_report(offset, offset1, saved_AI)

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
