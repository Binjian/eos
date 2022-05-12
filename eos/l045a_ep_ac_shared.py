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

This script shows an implementation of DDPG method on l045a truck real environment.

### Deep Deterministic Policy Gradient (DDPG) 

### Gym-Carla env 

An Ego Vehicle drives through a fixed track and collect loss (negative reward) defined
as energy consumption 

### References

- [DDPG ](https://keras.io/examples/rl/ddpg_pendulum/)

"""
"""
## Setup
"""

# drl import
import datetime

# from birdseye import eye

# from viztracer import VizTracer
# from watchpoints import watch

from collections import deque
from pythonjsonlogger import jsonlogger

# Logging Service Initialization
import logging
from logging.handlers import SocketHandler
import inspect


# resumption settings
parser = argparse.ArgumentParser(
    "use ddpg episodefree mode with tensorflow backend for VEOS with coastdown activated and expected velocity in 3 seconds"
)
parser.add_argument(
    "-r",
    "--resume",
    help="resume the last training with restored model, checkpoint and pedal map",
    action="store_true",
)

parser.add_argument(
    "-i",
    "--infer",
    help="No model update and training. Only Inference",
    action="store_false",
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
    help="relative path to be saved, for create subfolder for different drivers",
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
    "%(asctime)s-%(name)s-%(levelname)s-%(module)s-%(threadName)s-%(funcName)s)-%(lineno)d): %(message)s"
)
json_file_formatter = jsonlogger.JsonFormatter(
    "%(asctime)s %(name)s %(levelname)s %(module)s %(threadName)s %(funcName)s) %(lineno)d) %(message)s"
)
if args.path is None:
    args.path = "."
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
    "/l045a_ep_ac_shared-"
    + datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    + ".log"
)

fh = logging.FileHandler(logfilename)
fh.setLevel(logging.DEBUG)
fh.setFormatter(json_file_formatter)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
#  Cutelog socket
sh = SocketHandler("127.0.0.1", 19996)
sh.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)
logger.addHandler(sh)

logger.setLevel(logging.DEBUG)
# dictLogger = {'funcName': '__self__.__func__.__name__'}
# dictLogger = {'user': inspect.currentframe().f_back.f_code.co_name}
dictLogger = {"user": inspect.currentframe().f_code.co_name}

logc = logger.getChild("control flow")
logc.propagate = True
logd = logger.getChild("data flow")
logd.propagate = True

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


logger.info(f"Tensorflow Imported!", extra=dictLogger)

import socket
import json

# communication import
from threading import Lock, Thread
import queue, time, math, signal


# visualization import
import pandas as pd
import matplotlib.pyplot as plt
from visualization.visual import plot_to_image

logger.info(f"External Modules Imported!", extra=dictLogger)

# internal import
from comm import generate_vcu_calibration, set_tbox_sim_path, send_float_array
from agent import get_actor, get_critic, policy, Buffer, update_target, OUActionNoise

# set_tbox_sim_path("/home/veos/devel/newrizon/drl-carla-manual/src/comm/tbox")
set_tbox_sim_path(os.getcwd() + "/comm/tbox")
# value = [99.0] * 21 * 17
# send_float_array('TQD_trqTrqSetECO_MAP_v', value)

# DONE add vehicle communication interface
# DONE add model checkpoint episodically unique


# multithreading initialization
hmi_lock = Lock()


# DONE add visualization and logging
# Create folder for ckpts loggings.
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = datafolder + "/tf_logs/ddpg/gradient_tape/" + current_time + "/train"
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
action_budget = 0.10  # interval that allows modifying the calibration table
action_lower = 0.8
action_upper = 1.0
action_bias = 0.0

pd_index = np.linspace(0, 100, vcu_calib_table_row)
pd_index[1] = 7
pd_columns = (
    np.array([0, 2, 4, 8, 12, 16, 20, 24, 28, 32, 38, 44, 50, 62, 74, 86, 100]) / 100
)

target_velocity = np.array(
    [
        0,
        1.8,
        3.6,
        5.4,
        7.2,
        9,
        10.8,
        12.6,
        14.4,
        16.2,
        14.4,
        12.6,
        10.8,
        9,
        7.2,
        5.4,
        3.6,
        1.8,
        0,
        0,
        0,
    ]
)  # unit: km/h

pedal_range = [0, 1.0]
velocity_range = [0, 20.0]

# resume last pedal map / scratch from default table
if args.resume:
    vcu_calib_table0 = generate_vcu_calibration(
        vcu_calib_table_col,
        pedal_range,
        vcu_calib_table_row,
        velocity_range,
        3,
        datafolder,
    )
else:
    vcu_calib_table0 = generate_vcu_calibration(
        vcu_calib_table_col,
        pedal_range,
        vcu_calib_table_row,
        velocity_range,
        2,
        datafolder,
    )

vcu_calib_table1 = np.copy(vcu_calib_table0)  # shallow copy of the default table
vcu_table1 = vcu_calib_table1.reshape(-1).tolist()
logger.info(f"Start flash initial table", extra=dictLogger)
# time.sleep(1.0)
send_float_array("TQD_trqTrqSetNormal_MAP_v", vcu_table1, sw_diff=False)
logger.info(f"Done flash initial table", extra=dictLogger)
# TQD_trqTrqSetECO_MAP_v

# create actor-critic network
num_observations = 3  # observed are velocity, throttle, brake percentage; !! acceleration not available in l045a
sequence_len = 30  # 30 observation pairs as a valid observation for agent, for period of 50ms, this is equal to 1.5 second
num_inputs = num_observations * sequence_len  # 60 subsequent observations
num_actions = vcu_calib_table_size  # 17*21 = 357
vcu_calib_table_row_reduced = 5  # 0:5 adaptive rows correspond to low speed from  0~20, 7~25, 10~30, 15~35, etc  kmh  # overall action space is the whole table
num_reduced_actions = vcu_calib_table_row_reduced * vcu_calib_table_col  # 5x17=85
# hyperparameters for DRL
num_hidden = 256
num_hidden0 = 16
num_hidden1 = 32

# DYNAMIC: need to adapt the pointer to change different roi of the pm, change the starting row index
vcu_calib_table_row_start = 0
vcu_calib_table0_reduced = vcu_calib_table0[
    vcu_calib_table_row_start : vcu_calib_table_row_reduced + vcu_calib_table_row_start,
    :,
]

tf.keras.backend.set_floatx("float32")
# Initialize networks
actor_model = get_actor(
    num_observations,
    num_reduced_actions,
    sequence_len,
    num_hidden,
    action_bias,
)

critic_model = get_critic(
    num_observations,
    num_reduced_actions,
    sequence_len,
    num_hidden0,
    num_hidden1,
    num_hidden,
)


# Initialize networks
target_actor = get_actor(
    num_observations,
    num_reduced_actions,
    sequence_len,
    num_hidden,
    action_bias,
)

target_critic = get_critic(
    num_observations,
    num_reduced_actions,
    sequence_len,
    num_hidden0,
    num_hidden1,
    num_hidden,
)


# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005

buffer = Buffer(
    actor_model,
    critic_model,
    target_actor,
    target_critic,
    actor_optimizer,
    critic_optimizer,
    num_observations,
    sequence_len,
    num_reduced_actions,
    buffer_capacity=300000,
    batch_size=4,
    gamma=0.99,
    datafolder=datafolder,
)
# try buffer size with 1,000,000

# add checkpoints manager
if args.resume:
    checkpoint_actor_dir = datafolder + "/tf_ckpts-aa/l045a_ddpg_actor"
    checkpoint_critic_dir = datafolder + "/tf_ckpts-aa/l045a_ddpg_critic"
else:
    checkpoint_actor_dir = (
        datafolder
        + "/tf_ckpts-aa/l045a_ddpg_actor"
        + datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    )
    checkpoint_critic_dir = (
        datafolder
        + "/tf_ckpts-aa/l045a_ddpg_critic"
        + datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    )
try:
    os.makedirs(checkpoint_actor_dir)
    logger.info("Actor folder doesn't exist. Created!", extra=dictLogger)
except FileExistsError:
    logger.info("Actor folder exists, just resume!", extra=dictLogger)
try:
    os.makedirs(checkpoint_critic_dir)
    logger.info("User folder doesn't exist. Created!", extra=dictLogger)
except FileExistsError:
    logger.info("User folder exists, just resume!", extra=dictLogger)

ckpt_actor = tf.train.Checkpoint(
    step=tf.Variable(1), optimizer=actor_optimizer, net=actor_model
)
manager_actor = tf.train.CheckpointManager(
    ckpt_actor, checkpoint_actor_dir, max_to_keep=10
)
ckpt_actor.restore(manager_actor.latest_checkpoint)
if manager_actor.latest_checkpoint:
    logger.info(
        f"Actor Restored from {manager_actor.latest_checkpoint}", extra=dictLogger
    )
else:
    logger.info(f"Actor Initializing from scratch", extra=dictLogger)

ckpt_critic = tf.train.Checkpoint(
    step=tf.Variable(1), optimizer=critic_optimizer, net=critic_model
)
manager_critic = tf.train.CheckpointManager(
    ckpt_critic, checkpoint_critic_dir, max_to_keep=10
)
ckpt_critic.restore(manager_critic.latest_checkpoint)
if manager_critic.latest_checkpoint:
    logger.info(
        f"Critic Restored from {manager_critic.latest_checkpoint}", extra=dictLogger
    )
else:
    logger.info("Critic Initializing from scratch", extra=dictLogger)

# Making the weights equal initially after checkpoints load
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())


# todo ignites manual loading of tensorflow library, to guarantee the real-time processing of first data in main thread
init_motionpower = np.random.rand(sequence_len, num_observations)
init_states = tf.convert_to_tensor(
    init_motionpower
)  # state must have 30 (speed, throttle, current, voltage) 5 tuple
init_states = tf.expand_dims(init_states, 0)  # motion states is 30*2 matrix

# noise is a row vector of num_actions dimension
std_dev = 0.2
ou_noise = OUActionNoise(
    mean=np.zeros(num_reduced_actions),
    std_deviation=float(std_dev) * np.ones(num_reduced_actions),
)

action0 = policy(actor_model, init_states, ou_noise)
logger.info(f"manual load tf library by calling convert_to_tensor", extra=dictLogger)
ou_noise.reset()

# @eye
# tracer.start()
logger.info(f"Global Initialization done!", extra=dictLogger)


# tableQueue contains a table which is a list of type float
tableQueue = queue.Queue()
# figQueue is for visualization thread to show the calibration 3d figure
figQueue = queue.Queue()
# motionpowerQueue contains a vcu states list with N(20) subsequent motion states + reward as observation
motionpowerQueue = queue.Queue()


# initial status of the switches
program_exit = False
episode_done = False
episode_end = False
episode_count = 0


def reset_capture_handler():
    global program_exit
    get_truck_status.start = False
    logger.info(f"reset_capture_handler called", extra=dictLogger)
    raise Exception("reset capture to stop")


signal.signal(signal.SIGALRM, reset_capture_handler)


def get_truck_status():
    global program_exit
    global motionpowerQueue, sequence_len
    global episode_count, episode_done, episode_end
    global vcu_calib_table_row_start

    # logger.info(f'Start Initialization!', extra=dictLogger)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    socket.socket.settimeout(s, None)
    s.bind((get_truck_status.myHost, get_truck_status.myPort))
    # s.listen(5)
    # datetime.datetime.now().strftime("%Y%b%d-%H%M%S")
    start_moment = time.time()
    th_exit = False
    last_moment = time.time()
    logc.info(f"Initialization Done!", extra=dictLogger)
    # qobject_size = 0

    vel_hist_dQ = deque(maxlen=20)  # accumulate 1s of velocity values
    # vel_cycle_dQ = deque(maxlen=30)  # accumulate 1.5s (one cycle) of velocity values
    vel_cycle_dQ = deque(
        maxlen=sequence_len
    )  # accumulate 1.5s (one cycle) of velocity values

    while not th_exit:  # th_exit is local; program_exit is global
        with hmi_lock:  # wait for tester to kick off or to exit
            if program_exit == True:  # if program_exit is True, exit thread
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
        data_type = type(pop_data)
        logc.info(f"Data type is {data_type}", extra=dictLogger)
        if not isinstance(pop_data, dict):
            logd.critical(f"udp sending wrong data type!", extra=dictLogger)
            raise TypeError("udp sending wrong data type!")

        epi_delay_stop = False
        for key, value in pop_data.items():
            if key == "status":  # state machine chores
                # print(candata)
                if value == "begin":
                    get_truck_status.start = True
                    logc.info("%s", "Episode will start!!!", extra=dictLogger)
                    th_exit = False
                    # ts_epi_start = time.time()

                    vel_hist_dQ.clear()
                    epi_delay_stop = False
                    with hmi_lock:
                        episode_done = False
                        episode_end = False

                elif value == "end_valid":
                    # DONE for valid end wait for another 2 queue objects (3 seconds) to get the last reward!
                    # cannot sleep the thread since data capturing in the same thread, use signal alarm instead
                    get_truck_status.start = (
                        True  # do not stopping data capture immediately
                    )
                    get_truck_status.motpow_t = []
                    while not motionpowerQueue.empty():
                        motionpowerQueue.get()
                    logc.info("%s", "Episode done!!!", extra=dictLogger)
                    th_exit = False
                    vel_hist_dQ.clear()
                    epi_delay_stop = True
                    with hmi_lock:
                        episode_count += 1  # valid round increments
                        episode_done = True
                        episode_end = True
                elif value == "end_invalid":
                    get_truck_status.start = False
                    logc.info(f"Episode is interrupted!!!", extra=dictLogger)
                    get_truck_status.motpow_t = []
                    vel_hist_dQ.clear()
                    # motionpowerQueue.queue.clear()
                    # logc.info(
                    #     f"Episode motionpowerQueue has {motionpowerQueue.qsize()} states remaining",
                    #     extra=dictLogger,
                    # )
                    while not motionpowerQueue.empty():
                        motionpowerQueue.get()
                    # logc.info(
                    #     f"Episode motionpowerQueue gets cleared!", extra=dictLogger
                    # )
                    th_exit = False
                    epi_delay_stop = False
                    with hmi_lock:
                        episode_done = False
                        episode_end = True
                        episode_count += 1  # invalid round increments
                elif value == "exit":
                    get_truck_status.start = False
                    get_truck_status.motpow_t = []
                    vel_hist_dQ.clear()
                    while not motionpowerQueue.empty():
                        motionpowerQueue.get()
                    # logc.info("%s", "Program will exit!!!", extra=dictLogger)
                    th_exit = True
                    epi_delay_stop = False
                    # for program exit, need to set episode states
                    # final change to inform main thread
                    with hmi_lock:
                        episode_done = False
                        episode_end = True
                        program_exit = True
                        episode_count += 1
                    break
                    # time.sleep(0.1)
            elif key == "data":
                # logger.info('Data received before Capture starting!!!', extra=dictLogger)
                # logger.info(f'ts:{value["timestamp"]}vel:{value["velocity"]}ped:{value["pedal"]}', extra=dictLogger)
                # DONE add logic for episode valid and invalid
                if epi_delay_stop:
                    signal.alarm(3)  # delay stop for 3 seconds
                try:
                    if get_truck_status.start:  # starts episode

                        velocity = float(value["velocity"])
                        pedal = float(value["pedal"])
                        brake = float(value["brake_pressure"])
                        current = float(value["A"])
                        voltage = float(value["V"])

                        motion_power = [
                            velocity,
                            pedal,
                            brake,
                            current,
                            voltage,
                        ]  # 3 +2 : im 5

                        get_truck_status.motpow_t.append(
                            motion_power
                        )  # obs_reward [speed, pedal, brake, current, voltage]
                        vel_hist_dQ.append(velocity)
                        vel_cycle_dQ.append(velocity)

                        if len(get_truck_status.motpow_t) >= sequence_len:
                            if len(vel_cycle_dQ) != vel_cycle_dQ.maxlen:
                                logc.warning(  # the recent 1.5s average velocity
                                    f"cycle deque is inconsistent!",
                                    extra=dictLogger,
                                )

                            vel_aver = sum(vel_cycle_dQ) / vel_cycle_dQ.maxlen
                            vel_min = min(vel_cycle_dQ)
                            vel_max = max(vel_cycle_dQ)

                            # 0~20km/h; 7~25km/h; 10~30km/h; 15~35km/h; ...
                            # average concept
                            # 10; 16; 20; 25; 30; 35; 40; 45; 50; 55; 60;
                            #   13; 18; 22; 27; 32; 37; 42; 47; 52; 57; 62;
                            # here upper bound rule adopted
                            if vel_max < 20:
                                vcu_calib_table_row_start = 0
                            elif vel_max < 100:
                                vcu_calib_table_row_start = (
                                    math.floor((vel_max - 20) / 5) + 1
                                )
                            else:
                                logc.warning(
                                    f"cycle higher than 100km/h!",
                                    extra=dictLogger,
                                )
                                vcu_calib_table_row_start = 16

                            logd.info(
                                f"Cycle velocity: Aver{vel_aver},Min{vel_min},Max{vel_max},StartIndex{vcu_calib_table_row_start}!",
                                extra=dictLogger,
                            )
                            # logd.info(
                            #     f"Producer Queue has {motionpowerQueue.qsize()}!",
                            #     extra=dictLogger,
                            # )
                            motionpowerQueue.put(get_truck_status.motpow_t)
                            get_truck_status.motpow_t = []
                except Exception as X:
                    logc.info(
                        X,  # f"Valid episode, Reset data capturing to stop after 3 seconds!",
                        extra=dictLogger,
                    )
            else:
                logc.critical("udp sending unknown signal (neither status nor data)!")
                break

    logger.info(f"get_truck_status dies!!!", extra=dictLogger)

    s.close()


get_truck_status.motpow_t = []
get_truck_status.myHost = "127.0.0.1"
get_truck_status.myPort = 8002
get_truck_status.start = False
get_truck_status.qobject_len = 12  # sequence length 1.5*12s

# this is the calibration table consumer for flashing
# @eye
def flash_vcu(tablequeue):
    global program_exit

    flash_count = 0
    th_exit = False

    logc.info(f"Initialization Done!", extra=dictLogger)
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
            logc.info(f"flash starts", extra=dictLogger)
            send_float_array("TQD_trqTrqSetNormal_MAP_v", table, sw_diff=True)
            # time.sleep(1.0)
            logc.info(f"flash done, count:{flash_count}", extra=dictLogger)
            flash_count += 1
            # watch(flash_count)

    # motionpowerQueue.join()
    logc.info(f"flash_vcu dies!!!", extra=dictLogger)


# TODO add a thread for send_float_array
# TODO add printing calibration table
# TODO add initialize table to EP input
# @eye
def main():
    global episode_count
    global program_exit
    global motionpowerQueue
    global pd_index, pd_columns
    global episode_done, episode_end
    global vcu_calib_table_row_start

    eps = np.finfo(np.float32).eps.item()  # smallest number such that 1.0 + eps != 1.0

    # Start thread for flashing vcu, flash first
    thr_observe = Thread(target=get_truck_status, name="observe", args=())
    thr_flash = Thread(target=flash_vcu, name="flash", args=(tableQueue,))
    thr_observe.start()
    thr_flash.start()

    # todo connect gym-carla env, collect 20 steps of data for 1 second and update vcu calib table.

    """
    ## train
    """
    running_reward = 0
    episode_reward = 0
    th_exit = False
    epi_cnt_local = 0

    logger.info(f"main Initialization done!", extra=dictLogger)
    while not th_exit:  # run until solved or program exit; th_exit is local
        with hmi_lock:  # wait for tester to kick off or to exit
            th_exit = program_exit  # if program_exit is False,
            epi_cnt = episode_count  # get episode counts
            epi_end = episode_end
        if epi_end:  # if episode_end is True, wait for start of episode
            # logger.info(f'wait for start!', extra=dictLogger)
            continue

        step_count = 0
        wh1 = 0  # initialize odd step wh
        tf.summary.trace_on(graph=True, profiler=True)

        logc.info("----------------------", extra=dictLogger)
        logc.info(
            f"E{epi_cnt} starts!",
            extra=dictLogger,
        )
        with tf.GradientTape() as tape:
            while (
                not epi_end
            ):  # end signal, either the round ends normally or user interrupt
                # TODO l045a define round done (time, distance, defined end event)
                with hmi_lock:  # wait for tester to interrupt or to exit
                    th_exit = program_exit  # if program_exit is False, reset to wait
                    epi_end = episode_end
                    done = episode_done
                    table_start = vcu_calib_table_row_start

                if epi_end:  # stop observing and inferring
                    continue

                try:
                    logc.info(f"E{epi_cnt} Wait for an object!!!", extra=dictLogger)
                    motionpower = motionpowerQueue.get(block=True, timeout=1.55)
                except queue.Empty:
                    logc.info(
                        f"E{epi_cnt} No data in the Queue!!!",
                        extra=dictLogger,
                    )
                    continue

                logc.info(
                    f"E{epi_cnt} start step {step_count}",
                    extra=dictLogger,
                )  # env.step(action) action is flash the vcu calibration table
                # watch(step_count)
                # reward history
                motionpower_states = tf.convert_to_tensor(
                    motionpower
                )  # state must have 30 (velocity, pedal, brake, current, voltage) 5 tuple (num_observations)
                motion_states, power_states = tf.split(motionpower_states, [3, 2], 1)

                logd.info(
                    f"E{epi_cnt} tensor convert and split!",
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
                )  # vcu reward is a scalar
                wh = ui_sum / 3600.0 * 0.05  # negative wh
                # logger.info(
                #     f"ui_sum: {ui_sum}",
                #     extra=dictLogger,
                # )
                logd.info(
                    f"wh: {wh}",
                    extra=dictLogger,
                )

                if (
                    step_count % 2
                ) == 0:  # only for even observation/reward take an action
                    # k_cycle = 1000  # TODO determine the ratio
                    # cycle_reward += k_cycle * motion_magnitude.numpy()[0] # TODO add velocitoy sum as reward
                    # wh0 = wh  # add velocitoy sum as reward
                    cycle_reward = (wh1 + wh) * (
                        -1.0
                    )  # most recent odd and even indexed reward
                    episode_reward += cycle_reward
                    # TODO add speed sum as positive reward

                    if step_count != 0:
                        buffer.record(
                            (
                                prev_motion_states,
                                prev_action,
                                cycle_reward,
                                motion_states,
                            )
                        )
                    # motion_states_history.append(motion_states)
                    motion_states0 = motion_states

                    motion_states1 = tf.expand_dims(
                        motion_states0, 0
                    )  # motion states is 30*3 matrix

                    # predict action probabilities and estimated future rewards
                    # from environment state
                    # for causl rl, the odd indexed observation/reward are caused by last action
                    # skip the odd indexed observation/reward for policy to make it causal
                    logc.info(
                        f"E{epi_cnt} before inference!",
                        extra=dictLogger,
                    )
                    vcu_action_reduced = policy(actor_model, motion_states1, ou_noise)
                    prev_motion_states = motion_states0
                    prev_action = vcu_action_reduced

                    logd.info(
                        f"E{epi_cnt} inference done with reduced action space!",
                        extra=dictLogger,
                    )

                    vcu_calib_table_reduced = tf.reshape(
                        vcu_action_reduced,
                        [vcu_calib_table_row_reduced, vcu_calib_table_col],
                    )
                    # logger.info(
                    #     f"vcu action table reduced generated!", extra=dictLogger
                    # )
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

                    # dynamically change table row start index
                    vcu_calib_table0_reduced = vcu_calib_table0[
                        table_start : vcu_calib_table_row_reduced + table_start,
                        :,
                    ]
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
                        table_start : vcu_calib_table_row_reduced + table_start, :
                    ] = vcu_calib_table_reduced.numpy()
                    pds_curr_table = pd.DataFrame(
                        vcu_calib_table1, pd_index, pd_columns
                    )
                    # logc.info(
                    #     f"E{epi_cnt} start record instant table: {step_count}",
                    #     extra=dictLogger,
                    # )

                    if args.record_table:
                        curr_table_store_path = (
                            datafolder
                            + "/tables/instant_table_ddpg-bigep"
                            + datetime.datetime.now().strftime("%y-%m-%d-%h-%m-%s-")
                            + "e-"
                            + str(epi_cnt)
                            + "-"
                            + str(step_count)
                            + ".csv"
                        )
                        with open(curr_table_store_path, "wb") as f:
                            pds_curr_table.to_csv(curr_table_store_path)
                            # np.save(last_table_store_path, vcu_calib_table1)
                        last_table_store_path = os.getcwd() + "/../data/last_table.csv"
                        logd.info(
                            f"E{epi_cnt} done with record instant table: {step_count}",
                            extra=dictLogger,
                        )

                    vcu_act_list = vcu_calib_table1.reshape(-1).tolist()
                    # tf.print('calib table:', vcu_act_list, output_stream=sys.stderr)
                    tableQueue.put(vcu_act_list)
                    logd.info(
                        f"E{epi_cnt}StartIndex{table_start} Action Push table: {tableQueue.qsize()}",
                        extra=dictLogger,
                    )
                    logc.info(
                        f"E{epi_cnt} Finish Step: {step_count}",
                        extra=dictLogger,
                    )

                # during odd steps, old action remains effective due to learn and flash delay
                # so ust record the reward history
                # motion states (observation) are not used later for backpropagation
                else:
                    # cycle_reward = (wh0 + wh) * (-1.0)  # odd + even indexed reward
                    # Bugfix: the reward recorded in the first even step is not causal
                    # cycle_reward should include the most recent even wh in the even step
                    # record the odd step wh
                    wh1 = wh

                    # TODO add speed sum as positive reward
                    logc.info(
                        f"E{epi_cnt} Step done: {step_count}",
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
                logc.info(
                    f"E{epi_cnt} interrupted, waits for next episode to kick off!",
                    extra=dictLogger,
                )
                episode_reward = 0.0
                continue  # otherwise assuming the history is valid and back propagate

            logc.info(
                f"E{epi_cnt} Experience Collection ends!",
                extra=dictLogger,
            )

            if args.infer:
                (critic_loss, actor_loss) = buffer.nolearn()
                logd.info("No Learning, just calculating loss")
            else:
                for k in range(6):
                    # logger.info(f"BP{k} starts.", extra=dictLogger)
                    (critic_loss, actor_loss) = buffer.learn()
                    logd.info("Learning and updating")

                    update_target(target_actor.variables, actor_model.variables, tau)
                    # logger.info(f"Updated target actor", extra=dictLogger)
                    update_target(target_critic.variables, critic_model.variables, tau)
                    # logger.info(f"Updated target critic.", extra=dictLogger)

                # Checkpoint manager save model
                ckpt_actor.step.assign_add(1)
                ckpt_critic.step.assign_add(1)
                if int(ckpt_actor.step) % 5 == 0:
                    save_path_actor = manager_actor.save()
                    logd.info(
                        f"Saved checkpoint for step {int(ckpt_actor.step)}: {save_path_actor}",
                        extra=dictLogger,
                    )
                if int(ckpt_critic.step) % 5 == 0:
                    save_path_critic = manager_critic.save()
                    logd.info(
                        f"Saved checkpoint for step {int(ckpt_actor.step)}: {save_path_critic}",
                        extra=dictLogger,
                    )

            # logd.info(f"BP{k} done.", extra=dictLogger)
            logd.info(
                f"E{epi_cnt}BP{k} critic loss: {critic_loss}; actor loss: {actor_loss}",
                extra=dictLogger,
            )

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

        # tf logging after episode ends
        # use local episode counter epi_cnt_local tf.summary.writer; otherwise specify multiple logdir and automatic switch
        with train_summary_writer.as_default():
            tf.summary.scalar("WH", -episode_reward, step=epi_cnt_local)
            tf.summary.scalar("actor loss", actor_loss, step=epi_cnt_local)
            tf.summary.scalar("critic loss", critic_loss, step=epi_cnt_local)
            tf.summary.scalar("reward", episode_reward, step=epi_cnt_local)
            tf.summary.scalar("running reward", running_reward, step=epi_cnt_local)
            tf.summary.image(
                "Calibration Table", plot_to_image(fig), step=epi_cnt_local
            )
            tf.summary.histogram(
                "Calibration Table Hist", vcu_act_list, step=epi_cnt_local
            )
            # tf.summary.trace_export(
            #     name="veos_trace", step=epi_cnt_local, profiler_outdir=train_log_dir
            # )

        epi_cnt_local += 1
        plt.close(fig)

        logd.info(
            f"E{epi_cnt} Episode Reward: {episode_reward}",
            extra=dictLogger,
        )

        episode_reward = 0
        logc.info(
            f"E{epi_cnt} done, waits for next episode to kick off!",
            extra=dictLogger,
        )
        logc.info("----------------------", extra=dictLogger)
        if epi_cnt % 10 == 0:
            logc.info("++++++++++++++++++++++++", extra=dictLogger)
            logc.info(
                f"Running reward: {running_reward:.2f} at E{epi_cnt}",
                extra=dictLogger,
            )
            logc.info("++++++++++++++++++++++++", extra=dictLogger)

        # TODO terminate condition to be defined: reward > limit (percentage); time too long
    with train_summary_writer.as_default():
        tf.summary.trace_export(
            name="veos_trace", step=epi_cnt_local, profiler_outdir=train_log_dir
        )
    thr_observe.join()
    thr_flash.join()

    # TODOt  test restore last table
    logc.info(f"Save the last table!!!!", extra=dictLogger)

    pds_last_table = pd.DataFrame(vcu_calib_table1, pd_index, pd_columns)

    last_table_store_path = (
        datafolder  #  there's no slash in the end of the string
        + "/last_table_ddpg-"
        + datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        + ".csv"
    )
    with open(last_table_store_path, "wb") as f:
        pds_last_table.to_csv(last_table_store_path)
    buffer.save()

    logc.info(f"main dies!!!!", extra=dictLogger)


if __name__ == "__main__":
    main()
