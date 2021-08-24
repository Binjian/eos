"""
portNum = 8002  # port number
Title: Advantage Actor Critic Method
Author: [Binjian Xin](https://www.newrizon.com)
Date created: 2021/02/12
Last modified: 2020/03/15
Description: Implement Advantage Actor Critic Method in Carla environment.
"""
import sys
import os
import argparse

"""
## Introduction

This script shows an implementation of Advantage Actor Critic method on gym-carla environment.

### Actor Critic Method

As an agent takes actions and moves through an environment, it learns to map
the observed state of the environment to two possible outputs:

1. Recommended action: A probability value for each action in the action space.
   The part of the agent responsible for this output is called the **actor**.
2. Estimated rewards in the future: Sum of all rewards it expects to receive in the
   future. The part of the agent responsible for this output is the **critic**.

Agent and Critic learn to perform their tasks, such that the recommended actions
from the actor maximize the rewards.

### Gym-Carla env 

An Ego Vehicle drives through a fixed track and collect loss (negative reward) defined
as energy consumption 

### References

- [Actor Critic Method](https://hal.inria.fr/hal-00840470/document)

"""
"""
## Setup
"""

# drl import
import datetime
from birdseye import eye

# from viztracer import VizTracer
from watchpoints import watch


# Logging Service Initialization
import logging
import inspect


# resumption settings
parser = argparse.ArgumentParser("use a2c with tensorflow backend for VEOS with coastdown activated")
parser.add_argument(
    "-r",
    "--resume",
    help="resume the last training with restored model, checkpoint and pedal map",
    action="store_true",
)
parser.add_argument(
    "-t",
    "--record_table",
    help="record action table during training",
    action="store_true",
)
parser.add_argument(
    "-p",
    "--path",
    type=str,
    help="relative path to be saved, for create subfolder for different drivers"
)
args = parser.parse_args()


# tracer = VizTracer()
# logging.basicConfig(level=logging.DEBUG, format=fmt)
mpl_logger = logging.getLogger("matplotlib.font_manager")
mpl_logger.disabled = True

# logging.basicConfig(format=fmt)
logger = logging.getLogger("l045a")
logger.propagate = False
formatter = logging.Formatter(
    "%(asctime)s-%(levelname)s-%(module)s-%(threadName)s-%(funcName)s)-%(lineno)d): %(message)s"
)
if args.path is None:
    args.path = '.'
if args.resume:
    datafolder = "../data/" + args.path
else:
    datafolder = "../data/scratch/" + args.path

logfolder = datafolder + "/py_logs"
try:
    os.makedirs(logfolder)
except FileExistsError:
    print("User folder exists, just resume!")

logfilename = logfolder + (
        "/l045a_ac_tf-coastdown-"
        + datetime.datetime.now().strftime("%y-%m-%d-%h-%m-%s_%f")[:-3]
        + ".log"
)

fh = logging.FileHandler(logfilename)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

logger.setLevel(logging.DEBUG)
# dictLogger = {'funcName': '__self__.__func__.__name__'}
# dictLogger = {'user': inspect.currentframe().f_back.f_code.co_name}
dictLogger = {"user": inspect.currentframe().f_code.co_name}
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logger.info(f"Start Logging", extra=dictLogger)

if args.resume:
    logger.info(f"Resume last training", extra=dictLogger)
else:
    logger.info(f"Start from scratch", extra=dictLogger)


import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

from tensorflow.python.client import device_lib

logger.info(
    f"tensorflow device lib:\n{device_lib.list_local_devices()}\n", extra=dictLogger
)

from tensorflow import keras
import tensorflow_probability as tfp

tfd = tfp.distributions

logger.info(f"Tensorflow Imported!", extra=dictLogger)

import socket
import json

# communication import
from threading import Lock, Thread
import _thread as thread
import queue, time

# visualization import
import pandas as pd
import matplotlib.pyplot as plt
from visualization.visual import plot_to_image

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import io  # needed by convert figure to png in memory

logger.info(f"External Modules Imported!", extra=dictLogger)

# internal import
from comm.vcu_calib_generator import (
    generate_vcu_calibration,
    generate_lookup_table,
)


# from communication import carla_ros
from agent.ac_gaussian import (
    constructactorcriticnetwork,
    train_step,
)
from comm.tbox.scripts.tbox_sim import *

# set_tbox_sim_path("/home/veos/devel/newrizon/drl-carla-manual/src/comm/tbox")
set_tbox_sim_path(os.getcwd() + "/comm/tbox")
# value = [99.0] * 21 * 17
# send_float_array('TQD_trqTrqSetECO_MAP_v', value)

# TODO add vehicle communication insterface
# TODO add model checkpoint episodically unique


# multithreading initialization
vcu_step_lock = Lock()
hmi_lock = Lock()


# tableQueue contains a table which is a list of type float
tableQueue = queue.Queue()
# figQueue is for visualization thread to show the calibration 3d figure
figQueue = queue.Queue()
# motionpowerQueue contains a vcu states list with N(20) subsequent motion states + reward as observation
motionpowerQueue = queue.Queue()


# initial status of the switches
wait_for_reset = True
vcu_step = False
program_exit = False
episode_done = False
episode_count = 0
states_rewards = []

# TODO add visualization and logging
# Create folder for ckpts loggings.
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = datafolder + "/tf_logs/gradient_tape/" + current_time + "/train"
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
"""
## implement actor critic network

this network learns two functions:

1. actor: this takes as input the state of our environment and returns a
probability value for each action in its action space.
2. critic: this takes as input the state of our environment and returns
an estimate of total rewards in the future.

in our implementation, they share the initial layer.
"""
vcu_calib_table_col = 17  # number of pedal steps, x direction
vcu_calib_table_row = 21  # numnber of velocity steps, y direction
vcu_calib_table_budget = 0.05  # interval that allows modifying the calibration table
vcu_calib_table_size = vcu_calib_table_row * vcu_calib_table_col

pd_index = np.linspace(0, 100, vcu_calib_table_row)
pd_index[1] = 7
pd_columns = (
    np.array([0, 2, 4, 8, 12, 16, 20, 24, 28, 32, 38, 44, 50, 62, 74, 86, 100]) / 100
)

pedal_range = [0, 1.0]
velocity_range = [0, 20.0]

# resume last pedal map / scratch from default table
if args.resume:
    vcu_calib_table0 = generate_vcu_calibration(
        vcu_calib_table_col, pedal_range, vcu_calib_table_row, velocity_range, 3, datafolder
    )
else:
    vcu_calib_table0 = generate_vcu_calibration(
        vcu_calib_table_col, pedal_range, vcu_calib_table_row, velocity_range, 2, datafolder
    )

vcu_calib_table1 = np.copy(vcu_calib_table0)  # shallow copy of the default table
vcu_table1 = vcu_calib_table1.reshape(-1).tolist()
logger.info(f"Start flash initial table", extra=dictLogger)
# time.sleep(1.0)
send_float_array("TQD_trqTrqSetNormal_MAP_v", vcu_table1, sw_diff=False)
logger.info(f"Done flash initial table", extra=dictLogger)
# TQD_trqTrqSetECO_MAP_v

# create actor-critic network
num_observations = 2  # observed are the current speed and throttle; !! acceleration not available in l045a
sequence_len = 30  # 30 observation pairs as a valid observation for agent, for period of 50ms, this is equal to 1.5 second
num_inputs = num_observations * sequence_len  # 60 subsequent observations
num_actions = vcu_calib_table_size  # 17*21 = 357
vcu_calib_table_row_reduced = (
    5  # 1:5 rows correspond to low speed from  7, 10, 15, 20 kmh
)
num_reduced_actions = vcu_calib_table_row_reduced * vcu_calib_table_col  # 5x17=85
num_hidden = 128
bias_mu = 0.0  # bias 0.0 yields mu=0.0 with linear activation function
bias_sigma = 0.55  # bias 0.55 yields sigma=1.0 with softplus activation function

vcu_calib_table0_reduced = vcu_calib_table0[:vcu_calib_table_row_reduced, :]

tf.keras.backend.set_floatx("float64")
# TODO option fix sigma, just optimize mu
# TODO create testing scenes for gathering data
actorcritic_network = constructactorcriticnetwork(
    num_observations, sequence_len, num_reduced_actions, num_hidden, bias_mu, bias_sigma
)
gamma = 0.99  # discount factor for past rewards
opt = keras.optimizers.Adam(learning_rate=0.001)
# add checkpoints manager
if args.resume:
    checkpoint_dir = datafolder + "/tf_ckpts/l045a_ac_tf"
else:
    checkpoint_dir = (
            datafolder + "/tf_ckpts/l045a_ac_tf-"
            + datetime.datetime.now().strftime("%y-%m-%d-%h-%m-%s_%f")[:-3]
    )
try:
    os.makedirs(checkpoint_dir)
    logger.info("User folder doesn't exist. Created!", extra=dictLogger)
except FileExistsError:
    logger.info("User folder exists, just resume!", extra=dictLogger)

ckpt = tf.train.Checkpoint(
    step=tf.Variable(1), optimizer=opt, net=actorcritic_network
)
manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=10)
ckpt.restore(manager.latest_checkpoint)

if manager.latest_checkpoint:
    logger.info(f"Restored from {manager.latest_checkpoint}", extra=dictLogger)
else:
    logger.info(f"Initializing from scratch", extra=dictLogger)


# todo ignites manual loading of tensorflow library, to guarantee the real-time processing of first data in main thread
init_motionpower = np.random.rand(sequence_len, num_observations)
init_states = tf.convert_to_tensor(
    init_motionpower
)  # state must have 30 (speed, acceleration, throttle, current, voltage) 5 tuple
init_states = tf.expand_dims(init_states, 0)  # motion states is 30*2 matrix
mu_sigma, critic_value = actorcritic_network(init_states)
logger.info(f"manual load tf library by calling convert_to_tensor", extra=dictLogger)


# @eye
# tracer.start()
logger.info(f"Global Initialization done!", extra=dictLogger)


def get_truck_status():
    global episode_done, wait_for_reset, program_exit, motionpowerQueue, sequence_len
    # logger.info(f'Start Initialization!', extra=dictLogger)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    socket.socket.settimeout(s, None)
    s.bind((get_truck_status.myHost, get_truck_status.myPort))
    # s.listen(5)
    # datetime.datetime.now().strftime("%Y%b%d-%H%M%S")
    start_moment = time.time()
    th_exit = False
    last_moment = time.time()
    logger.info(f"Initialization Done!", extra=dictLogger)

    while not th_exit:
        with hmi_lock:  # This happens when policy turns deterministic, entropy goes to 0.
            if program_exit == True:  # if program_exit is True, exit thread.
                logger.info(
                    "%s",
                    "Capture thread exit due to processing request!!!",
                    extra=dictLogger,
                )
                th_exit = True
                continue
        candata, addr = s.recvfrom(2048)
        # logger.info('Data received!!!', extra=dictLogger)
        pop_data = json.loads(candata)
        if len(pop_data) != 1:
            logging.critical("udp sending multiple shots!")
            break
        for key, value in pop_data.items():
            if key == "status":  # state machine chores
                # print(candata)
                with hmi_lock:
                    if value == "begin":
                        get_truck_status.start = True
                        logger.info("%s", "Capture will start!!!", extra=dictLogger)
                        wait_for_reset = False  #  ignites the episode when tester kicks off; remains false within an episode
                        episode_done = False
                        program_exit = False
                        th_exit = False
                    elif (
                        value == "end_valid"
                    ):  # todo for valid end wait for another 2 queue objects (3 seconds) to get the last reward!
                        get_truck_status.start = False  # todo for the simple test case coast down is fixed. action cannot change the reward.
                        logger.info("%s", "Capture ends!!!", extra=dictLogger)
                        wait_for_reset = True  # wait when episode starts
                        episode_done = True
                        program_exit = False
                        th_exit = False
                    elif value == "end_invalid":
                        get_truck_status.start = False
                        logger.info("%s", "Capture is interrupted!!!", extra=dictLogger)
                        wait_for_reset = True  # wait when episode starts
                        episode_done = False
                        program_exit = False
                        th_exit = False
                    elif value == "exit":
                        get_truck_status.start = False
                        logger.info("%s", "Capture will exit!!!", extra=dictLogger)
                        wait_for_reset = (
                            True  # reset main thread to recheck exit status
                        )
                        episode_done = False
                        program_exit = True
                        th_exit = True
                        break
                        # time.sleep(0.1)
            elif key == "data":
                # logger.info('Data received before Capture starting!!!', extra=dictLogger)
                # logger.info(f'ts:{value["timestamp"]}vel:{value["velocity"]}ped:{value["pedal"]}', extra=dictLogger)

                if get_truck_status.start:
                    timestamp = float(value["timestamp"])
                    velocity = float(value["velocity"])
                    # acceleration in invalid for l045a
                    acceleration = float(value["acceleration"])
                    pedal = float(value["pedal"])
                    current = float(value["A"])
                    voltage = float(value["V"])
                    power = float(value["W"])
                    motion_power = [velocity, pedal, current, voltage]
                    step_moment = time.time()
                    # step_dt_object = datetime.datetime.fromtimestamp(step_moment)
                    # send_moment = float(timestamp) / 1e06 - 28800
                    # send_dt_object = datetime.datetime.fromtimestamp(send_moment)

                    # logger.info(f'transmission delay: {step_moment-send_moment:6f}', extra=dictLogger)
                    # logger.info(f'step interval:{step_moment-last_moment :6f}', extra=dictLogger)
                    last_moment = step_moment
                    # time.sleep(0.1)
                    # print("timestamp:{},velocity:{},acceleration:{},pedal:{},current:{},voltage:{},power:{}".format(timestamp,velocity,acceleration,pedal,current,voltage,power))
                    get_truck_status.motionpower_states.append(
                        motion_power
                    )  # obs_reward [speed, pedal, current, voltage]
                    # logger.info(f'motionpower: {motion_power}', extra=dictLogger)
                    if len(get_truck_status.motionpower_states) >= sequence_len:
                        # print(f"motion_power num: {len(get_truck_status.motionpower_states)}")
                        # empty queues for main thread to get the most fresh motion power state
                        # logger.info(
                        #     f"Producer creates {motionpowerQueue.qsize()}",
                        #     extra=dictLogger,
                        # )
                        logger.info(
                            f"Producer Queue has {motionpowerQueue.qsize()}!",
                            extra=dictLogger,
                        )
                        # if not motionpowerQueue.empty():
                        #     logger.info(
                        #         f"Producer Queue has {motionpowerQueue.qsize()}, will be cleaned up!",
                        #         extra=dictLogger,
                        #     )
                        # else:
                        #     logger.info(
                        #         f"Producer Queue is clean!",
                        #         extra=dictLogger,
                        #     )
                        # while not motionpowerQueue.empty():
                        #     motionpowerQueue.get()

                        motionpowerQueue.put(get_truck_status.motionpower_states)
                        # watch(motionpowerQueue.qsize())

                        # motionpower_states_s = [f"{vel:.3f},{ped:.3f},{c:.3f},{v:.3f}" for (vel, ped, c, v) in get_truck_status.motionpower_states]
                        # logger.info(
                        #     f"Motion Power States: {motionpower_states_s}",
                        #     extra=dictLogger,
                        # )
                        logger.info(
                            "Motion Power States put in Queue!!!", extra=dictLogger
                        )
                        get_truck_status.motionpower_states = []
            else:
                continue
        # time.sleep(0.1)
    # motionpowerQueue.join()
    logger.info(f"get_truck_status dies!!!", extra=dictLogger)

    s.close()


get_truck_status.motionpower_states = []
get_truck_status.myHost = "127.0.0.1"
get_truck_status.myPort = 8002
get_truck_status.start = False


# this is the calibration table consumer for flashing
# @eye
def flash_vcu(tablequeue):
    global program_exit

    flash_count = 0
    th_exit = False

    logger.info(f"Initialization Done!", extra=dictLogger)
    while not th_exit:
        # time.sleep(0.1)
        with hmi_lock:
            if program_exit:
                th_exit = True
                continue
        try:
            # print("1 tablequeue size: {}".format(tablequeue.qsize()))
            table = tablequeue.get(block=False, timeout=1)  # default block = True
            # print("2 tablequeue size: {}".format(tablequeue.qsize()))
        except queue.Empty:
            pass
        else:

            # tf.print('calib table:', table, output_stream=output_path)
            logger.info(f"flash starts", extra=dictLogger)
            send_float_array("TQD_trqTrqSetNormal_MAP_v", table, sw_diff=True)
            # time.sleep(1.0)
            logger.info(f"flash count:{flash_count}", extra=dictLogger)
            flash_count += 1
            # watch(flash_count)

    # motionpowerQueue.join()
    logger.info(f"flash_vcu dies!!!", extra=dictLogger)


# TODO add a thread for send_float_array
# TODO add printing calibration table
# TODO add initialize table to EP input
# @eye
def main():
    global episode_done, episode_count, wait_for_reset, program_exit
    global states_rewards
    global vcu_step
    global motionpowerQueue

    eps = np.finfo(np.float64).eps.item()  # smallest number such that 1.0 + eps != 1.0

    # Start thread for flashing vcu, flash first
    thr_observe = Thread(target=get_truck_status, name="observe", args=())
    thr_flash = Thread(target=flash_vcu, name="flash", args=(tableQueue,))
    thr_observe.start()
    thr_flash.start()

    # todo connect gym-carla env, collect 20 steps of data for 1 second and update vcu calib table.

    """
    ## train
    """
    vcu_action_history = []
    mu_sigma_history = []
    vcu_critic_value_history = []
    vcu_rewards_history = []
    running_reward = 0
    episode_reward = 0
    motion_states_history = []
    th_exit = False
    done = False
    episode_end = False

    logger.info(f"main Initialization done!", extra=dictLogger)
    while not th_exit:  # run until solved or program exit
        with hmi_lock:  # wait for tester to kick off or to exit
            th_exit = program_exit  # if program_exit is false, reset to wait
            if wait_for_reset:  # if program_exit is true, reset to exit
                # logger.info(f'wait for start!', extra=dictLogger)
                continue
            else:
                episode_end = False  # kick off

        step_count = 0
        with tf.GradientTape() as tape:
            while (
                not episode_end
            ):  # end signal, either the episode ends normally or user interrupt
                # TODO l045a define episode done (time, distance, defined end event)
                # obs, r, done, info = env.step(action)
                # episode_done = done
                with hmi_lock:  # wait for tester to interrupt or to exit
                    th_exit = program_exit  # if program_exit is false, reset to wait
                    episode_end = wait_for_reset
                    done = episode_done

                if episode_end and done:
                    logger.info(
                        f"Episode {episode_count+1} Experience Collection ends!",
                        extra=dictLogger,
                    )
                    continue
                elif episode_end and (not done):
                    logger.info(
                        f"Episode {episode_count+1} Experience Collection is interrupted!",
                        extra=dictLogger,
                    )
                    continue

                try:
                    logger.warning(f"Wait for an object!!!", extra=dictLogger)
                    motionpower = motionpowerQueue.get(block=True, timeout=1.55)
                except queue.Empty:
                    logger.warning(f"No data in the Queue!!!", extra=dictLogger)
                    continue

                logger.info(
                    f"Episode {episode_count + 1} start step {step_count}",
                    extra=dictLogger,
                )  # env.step(action) action is flash the vcu calibration table
                # watch(step_count)
                # reward history
                motionpower_states = tf.convert_to_tensor(
                    motionpower
                )  # state must have 30 (velocity, pedal, current, voltage) 4 tuple
                motion_states, power_states = tf.split(motionpower_states, [2, 2], 1)

                logger.info(
                    f"Episode {episode_count+1} tensor convert and split!",
                    extra=dictLogger,
                )
                # motion_states_s = [f"{vel:.3f},{ped:.3f}" for (vel, ped) in motion_states]
                # logger.info(
                #     f"Motion States: {motion_states_s}",
                #     extra=dictLogger,
                # )
                # power_states_s = [f"{c:.3f},{v:.3f}" for (c, v) in power_states]
                # logger.info(
                #     f"Power States: {power_states_s}",
                #     extra=dictLogger,
                # )
                # rewards should be a 20x2 matrix after split
                # reward is sum of power (U*I)
                ui_sum = tf.reduce_sum(
                    tf.reduce_prod(power_states, 1)
                )  # vcu_reward is a scalar
                wh = ui_sum / 3600.0 * 0.05  # negative wh
                logger.info(
                    f"ui_sum: {ui_sum}",
                    extra=dictLogger,
                )
                logger.info(
                    f"wh: {wh}",
                    extra=dictLogger,
                )

                if (
                    step_count % 2
                ) == 0:  # only for even observation/reward take an action
                    # k_vcu_reward = 1000  # TODO determine the ratio
                    # vcu_reward += k_vcu_reward * motion_magnitude.numpy()[0] # TODO add velocitoy sum as reward
                    wh0 = wh  # add velocitoy sum as reward
                    # TODO add speed sum as positive reward

                    # motion_states_history.append(motion_states)
                    motion_states0 = motion_states

                    motion_states0 = tf.expand_dims(
                        motion_states0, 0
                    )  # motion states is 30*2 matrix

                    # predict action probabilities and estimated future rewards
                    # from environment state
                    # for causl rl, the odd indexed observation/reward are caused by last action
                    # skip the odd indexed observation/reward for policy to make it causal
                    logger.info(
                        f"Episode {episode_count+1} before inference!", extra=dictLogger
                    )
                    # Now with reduced action space (only low speed, 5 rows)
                    mu_sigma, critic_value = actorcritic_network(motion_states0)

                    logger.info(
                        f"Episode {episode_count+1} inference done with reduced action space!",
                        extra=dictLogger,
                    )
                    vcu_critic_value_history.append(critic_value[0, 0])
                    mu_sigma_history.append(mu_sigma)

                    logger.info(f"mu sigma appended!", extra=dictLogger)
                    # mu_sigma_list = tf.transpose(tf.squeeze(mu_sigma))
                    # mu_sigma_s = [f"{mu:.3f},{sigma:.3f}" for (mu, sigma) in mu_sigma_list]
                    # logger.info(
                    #     f"mu sigma: {mu_sigma_s}",
                    #     extra=dictLogger,
                    # )

                    # sample action from action probability distribution
                    nn_mu, nn_sigma = tf.unstack(mu_sigma)
                    mvn = tfd.MultivariateNormalDiag(loc=nn_mu, scale_diag=nn_sigma)
                    vcu_action_reduced = mvn.sample()  # 17*4 = 68 actions
                    logger.info(
                        f"Episode {episode_count+1} Step {step_count} sampling done!",
                        extra=dictLogger,
                    )
                    vcu_action_history.append(vcu_action_reduced)
                    # Here the lookup table with constrained output is part of the environment,
                    # clip is part of the environment to be learned
                    # action is not constrained!
                    vcu_calib_table_reduced = tf.reshape(
                        vcu_action_reduced,
                        [vcu_calib_table_row_reduced, vcu_calib_table_col],
                    )
                    logger.info(
                        f"vcu action table reduced generated!", extra=dictLogger
                    )
                    # vcu_action_table_reduced_s = [f"{col:.3f},"
                    #                               for row in vcu_calib_table_reduced
                    #                               for col in row]
                    # logger.info(
                    #     f"vcu action table: {vcu_action_table_reduced_s}",
                    #     extra=dictLogger,
                    # )

                    # get change budget : % of initial table
                    vcu_calib_table_bound = 250
                    vcu_calib_table_reduced = (
                        vcu_calib_table_reduced * vcu_calib_table_bound
                    )
                    # vcu_calib_table_reduced = tf.math.multiply(
                    #     vcu_calib_table_reduced * vcu_calib_table_budget,
                    #     vcu_calib_table0_reduced,
                    # )
                    # add changes to the default value
                    # vcu_calib_table_min_reduced = 0.8 * vcu_calib_table0_reduced
                    vcu_calib_table_min_reduced = (
                        vcu_calib_table0_reduced - vcu_calib_table_bound
                    )
                    vcu_calib_table_max_reduced = 1.0 * vcu_calib_table0_reduced

                    vcu_calib_table_reduced = tf.clip_by_value(
                        vcu_calib_table_reduced + vcu_calib_table0_reduced,
                        clip_value_min=vcu_calib_table_min_reduced,
                        clip_value_max=vcu_calib_table_max_reduced,
                    )

                    # create updated complete pedal map, only update the first few rows
                    vcu_calib_table1[
                        :vcu_calib_table_row_reduced, :
                    ] = vcu_calib_table_reduced.numpy()
                    # vcu_calib_table1[:vcu_calib_table_row_reduced+1, :] = np.zeros( vcu_calib_table0_reduced.shape)
                    pds_curr_table = pd.DataFrame(
                        vcu_calib_table1, pd_index, pd_columns
                    )
                    logger.info(f"Episode {episode_count+1} Start record instant table: {step_count}", extra=dictLogger)

                    if args.record_table:
                        curr_table_store_path = (
                            os.getcwd()
                            + "/../data/scratch/instant_table"
                            + datetime.datetime.now().strftime("%y-%m-%d-%h-%m-%s_%f")[
                                :-3
                            ]
                            + str(episode_count + 1)
                            + "-"
                            + str(step_count)
                            + ".csv"
                        )
                        with open(curr_table_store_path, "wb") as f:
                            pds_curr_table.to_csv(curr_table_store_path)
                            # np.save(last_table_store_path, vcu_calib_table1)
                        last_table_store_path = os.getcwd() + "/../data/last_table.csv"
                    logger.info(f"Episode {episode_count+1} Done with record instant table: {step_count}", extra=dictLogger)

                    vcu_act_list = vcu_calib_table1.reshape(-1).tolist()
                    # tf.print('calib table:', vcu_act_list, output_stream=sys.stderr)
                    tableQueue.put(vcu_act_list)
                    logger.info(
                        f"Episode {episode_count+1} Action Push table: {tableQueue.qsize()}",
                        extra=dictLogger,
                    )
                    logger.info(
                        f"Episode {episode_count+1} Step done: {step_count}",
                        extra=dictLogger,
                    )
                # during odd steps, old action remains effective due to learn and flash delay
                # so ust record the reward history
                # motion states (observation) are not used later for backpropagation
                else:
                    vcu_reward = (wh0 + wh) * (-1.0)  # odd + even indexed reward
                    # TODO add speed sum as positive reward
                    vcu_rewards_history.append(vcu_reward)
                    episode_reward += vcu_reward
                    # episode_wh += wh  # if only the odd steps, then only considers the effective action results?
                    logger.info(
                        f"Episode {episode_count+1} Step done: {step_count}",
                        extra=dictLogger,
                    )

                    # motion_states = tf.stack([motion_states0, motion_states])
                    # motion_states_history was not used for back propagation
                    # 60 frames, but never used again
                    # motion_states_history.append(motion_states)

                # step level
                step_count += 1

            if (
                not done
            ):  # if user interrupt prematurely or exit, then ignore back propagation since data incomplete
                logger.info(
                    f"Episode {episode_count+1}  interrupted, waits for next episode kicking off!",
                    extra=dictLogger,
                )
                # clean up vcu_rewards_history, mu_sigma_history, episode_reward
                motion_states_history.clear()
                vcu_action_history.clear()
                vcu_rewards_history.clear()
                mu_sigma_history.clear()
                vcu_critic_value_history.clear()
                episode_reward = 0
                continue  # otherwise assuming the history is valid and back propagate

            # pm_output_path = "PMap-" + logfilename + f"_{episode_count}.out"
            # tf.print("calib table:", vcu_calib_table1, output_stream=pm_output_path)
            # Create a matplotlib 3d figure, //export and save in log
            pd_data = pd.DataFrame(
                vcu_calib_table1,
                columns=np.linspace(0, 1.0, num=17),
                index=np.linspace(0, 30, num=21),
            )
            df = pd_data.unstack().reset_index()
            df.columns = ["pedal", "velocity", "throttle"]

            fig = plt.figure()
            ax = plt.axes(projection="3d")
            surf = ax.plot_trisurf(
                df["pedal"],
                df["velocity"],
                df["throttle"],
                cmap=plt.cm.viridis,
                linewidth=0.2,
            )
            fig.colorbar(surf, shrink=0.5, aspect=5)
            ax.view_init(30, 135)
            # plt.show()
            # time.sleep(5)
            # update running reward to check condition for solving
            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

            # todo calculate return
            # calculate expected value from rewards
            # - at each timestep what was the total reward received after that timestep
            # - rewards in the past are discounted by multiplying them with gamma
            # - these are the labels for our critic
            returns = []
            discounted_sum = 0
            # everytime when episode ends, number of rewards items in history is always even
            # if action history is odd when episode ends, then the last action/mu_sigma is ignored for backpropagation
            # thus no extra handling is required
            # TODO move vcu_rewards one step forward !!!
            vcu_rewards_history = vcu_rewards_history[1:]
            vcu_action_history = vcu_action_history[:-1]
            mu_sigma_history = mu_sigma_history[:-1]
            vcu_critic_value_history = vcu_critic_value_history[:-1]

            for r in vcu_rewards_history[::-1]:
                discounted_sum = r + gamma * discounted_sum
                returns.insert(0, discounted_sum)
            # normalize
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            returns = returns.tolist()

            # calculating loss values to update our network
            history = zip(
                vcu_action_history, mu_sigma_history, vcu_critic_value_history, returns
            )

            logger.info(f"BP starts.", extra=dictLogger)
            # back propagation
            (
                loss_all,
                act_losses_all,
                entropy_losses_all,
                critic_losses_all,
            ) = train_step(actorcritic_network, history, opt, tape)
            logger.info(f"BP ends.", extra=dictLogger)
            ckpt.step.assign_add(1)

        with train_summary_writer.as_default():
            tf.summary.scalar("WH", -episode_reward, step=episode_count)
            tf.summary.scalar("loss_sum", loss_all, step=episode_count)
            tf.summary.scalar("loss_act", act_losses_all, step=episode_count)
            tf.summary.scalar("loss_entropy", entropy_losses_all, step=episode_count)
            tf.summary.scalar("loss_critic", critic_losses_all, step=episode_count)
            tf.summary.scalar("reward", episode_reward, step=episode_count)
            tf.summary.scalar("running reward", running_reward, step=episode_count)
            tf.summary.image(
                "Calibration Table", plot_to_image(fig), step=episode_count
            )
            tf.summary.histogram(
                "Calibration Table Hist", vcu_act_list, step=episode_count
            )
        plt.close(fig)

        logger.info(
            f"Episode {episode_count + 1}, Loss all: {loss_all}, Act loss: {act_losses_all}, Entropy loss: {entropy_losses_all}, Critic loss: {critic_losses_all}, Episode Reward: {episode_reward}",
            extra=dictLogger,
        )
        #
        #   # Reset metrics every epoch
        #   train_loss.reset_states()
        #   test_loss.reset_states()
        #   train_accuracy.reset_states()
        #   test_accuracy.reset_states()
        #   clear the loss and reward history
        motion_states_history.clear()
        vcu_action_history.clear()
        vcu_rewards_history.clear()
        mu_sigma_history.clear()
        vcu_critic_value_history.clear()
        # obs = env.reset()

        # log details
        # actorcritic_network.save_weights("./checkpoints/cp-{epoch:04d}.ckpt")
        # actorcritic_network.save("./checkpoints/cp-last.kpt")
        # ckp_moment = datetime.datetime.now().strftime("%Y%b%d-%H%M%S")
        # last_model_save_path = f"./checkpoints/cp-{ckp_moment}-{episode_count}.ckpt"
        # actorcritic_network.save(last_model_save_path)

        # Checkpoint manager save model
        save_path = manager.save()
        logger.info(
            f"Saved checkpoint for step {int(ckpt.step)}: {save_path}", extra=dictLogger
        )
        logger.info(
            f"loss {loss_all.numpy():.2f}".format(loss_all.numpy()), extra=dictLogger
        )

        episode_count += 1
        episode_reward = 0
        if episode_count % 10 == 0:
            logger.info("========================", extra=dictLogger)
            logger.info(
                f"running reward: {running_reward:.2f} at episode {episode_count}",
                extra=dictLogger,
            )

        if entropy_losses_all < 1e-10:
            logger.info(
                f"Policy becomes deterministic. Training ends due to local optimum."
            )
            th_exit = True
            with hmi_lock:  # wait for tester to kick off or to exit
                program_exit = True  # if program_exit is false, reset to wait

        logger.info(
            f"Episode {episode_count} done, waits for next episode kicking off!",
            extra=dictLogger,
        )
        # TODO terminate condition to be defined: reward > limit (percentage); time too long
        # if running_reward > 195:  # condition to consider the task solved
        #     print("solved at episode {}!".format(episode_count))
        #     break

    thr_observe.join()
    thr_flash.join()

    # todo test restore last table
    logger.info(f"Save the last table!!!!", extra=dictLogger)
    # cpypath = os.getcwd() + '../data/last_table.json'
    # copyfile("/dev/shm/out.json", cpypat)
    # pd_index = np.linspace(0, 100, vcu_calib_table_row)
    # pd_index[1] = 7
    # pd_columns = np.array([0, 2, 4, 8, 12, 16, 20, 24, 28, 32, 38, 44, 50, 62, 74, 86, 100]) / 100
    # columns = np.arange(17)
    # index = np.arrange(21)

    pds_last_table = pd.DataFrame(vcu_calib_table1, pd_index, pd_columns)

    last_table_store_path = (
        datafolder  #  there's no slash in the end of the string
        + "/last_table"
        + datetime.datetime.now().strftime("%y-%m-%d-%h-%m-%s_%f")[:-3]
        + ".csv"
    )
    with open(last_table_store_path, "wb") as f:
        pds_last_table.to_csv(last_table_store_path)

    # this is not needed since when initialization we get the latest table by timestamp instead of name.
    # if args.resume:
    #     resume_table_store_path = datafolder + "last_table.csv"
    #     with open(resume_table_store_path, "wb") as f:
    #         pds_last_table.to_csv(resume_table_store_path)

    logger.info(f"main dies!!!!", extra=dictLogger)


if __name__ == "__main__":
    main()
