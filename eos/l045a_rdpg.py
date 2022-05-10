"""
portNum = 8002  # port number
Title: Advantage Actor Critic Method
Author: [Binjian Xin](https://www.newrizon.com)
Date created: 2021/02/12
Last modified: 2020/03/15
Description: Implement Advantage Actor Critic Method in Carla environment.
"""


"""
## Introduction

This script shows an implementation of RDPG method on l045a truck real environment.

### Deep Deterministic Policy Gradient (RDPG) 

### Gym-Carla env 

An Ego Vehicle drives through a fixed track and collect loss (negative reward) defined
as energy consumption 

### References

- [RDPG ](https://keras.io/examples/rl/rdpg_pendulum/)

"""
"""
## Setup
"""

# system import
import sys
import os
import argparse
import logging

import datetime
from collections import deque

# communication import
from threading import Lock, Thread
import queue, time, math, signal

# third party import

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

# from birdseye import eye
# from viztracer import VizTracer
# from watchpoints import watch

# visualization import
import pandas as pd
import matplotlib.pyplot as plt
from visualization.visual import plot_to_image

# local import
from comm import generate_vcu_calibration, RemoteCan, send_float_array
from utils import get_logger, get_truck_status, flash_vcu, plot_3d_figure
from agent import RDPG


# resumption settings
parser = argparse.ArgumentParser(
    "use rdpg episodefree mode with tensorflow backend for VEOS with coastdown activated and expected velocity in 3 seconds"
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


if args.path is None:
    args.path = "."
if args.resume:
    datafolder = "../data/" + args.path
else:
    datafolder = (
        "../data/scratch/"
        + datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        + args.path
    )

logfolder = datafolder + "/py_logs"
try:
    os.makedirs(logfolder)
except FileExistsError:
    print("User folder exists, just resume!")
logger, dictLogger = get_logger(logfolder, "l045a_rdpg", logging.DEBUG)


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


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)


logger.info(
    f"tensorflow device lib:\n{device_lib.list_local_devices()}\n", extra=dictLogger
)

logger.info(f"Tensorflow Imported!", extra=dictLogger)


logger.info(f"External Modules Imported!", extra=dictLogger)


# set_tbox_sim_path("/home/veos/devel/newrizon/drl-carla-manual/src/comm/tbox")
# value = [99.0] * 21 * 17
# send_float_array('TQD_trqTrqSetECO_MAP_v', value)

# DONE add vehicle communication interface
# DONE add model checkpoint episodically unique


# multithreading initialization
hmi_lock = Lock()


# DONE add visualization and logging
# Create folder for ckpts loggings.
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = datafolder + "/tf_logs/rdpg/gradient_tape/" + current_time + "/train"
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
obs_len = 30  # 30 observation pairs as a valid observation for agent, for period of 50ms, this is equal to 1.5 second
num_inputs = num_observations * obs_len  # 30 subsequent observations
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
# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005
seq_len = 6  # TODO  7 maximum sequence length

# add checkpoints manager


rdpg = RDPG(
    num_observations,
    obs_len,
    seq_len,
    num_reduced_actions,
    buffer_capacity=300000,
    batch_size=4,
    hidden_unitsAC=(256, 256),
    n_layersAC=(2, 2),
    padding_value=-10000,  # padding value for the input, impossible value for observation, action or reward
    gamma=gamma,
    tauAC=(0.001, 0.001),
    lrAC=(0.001, 0.002),
    datafolder=datafolder,
    ckpt_interval=5,
)
# try buffer size with 1,000,000

# todo ignites manual loading of tensorflow library, to guarantee the real-time processing of first data in main thread
init_motionpower = np.random.rand(obs_len, num_observations)
init_states = tf.convert_to_tensor(
    init_motionpower
)  # state must have 30 (speed, throttle, current, voltage) 5 tuple
init_states = tf.expand_dims(init_states, 0)  # motion states is 30*2 matrix

# action0 = policy(actor_model, init_states, ou_noise)
action0 = rdpg.actor_predict(init_states, 0)

logger.info(f"manual load tf library by calling convert_to_tensor", extra=dictLogger)
rdpg.reset_noise()

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

    # eps = np.finfo(np.float32).eps.item()  # smallest number such that 1.0 + eps != 1.0

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
            motpow_t = tf.convert_to_tensor(
                motionpower
            )  # state must have 30 (velocity, pedal, brake, current, voltage) 5 tuple (num_observations)
            o_t, pow_t = tf.split(motpow_t, [3, 2], 1)

            logd.info(
                f"E{epi_cnt} tensor convert and split!",
                extra=dictLogger,
            )
            # o_t_s = [f"{vel:.3f},{ped:.3f}" for (vel, ped) in o_t]
            # logger.info(
            #     f"Motion States: {o_t_s}",
            #     extra=dictLogger,
            # )
            # pow_t_s = [f"{c:.3f},{v:.3f}" for (c, v) in pow_t]
            # logger.info(
            #     f"Power States: {pow_t_s}",
            #     extra=dictLogger,
            # )
            # rewards should be a 20x2 matrix after split
            # reward is sum of power (U*I)
            ui_sum = tf.reduce_sum(tf.reduce_prod(pow_t, 1))  # vcu reward is a scalar
            wh = ui_sum / 3600.0 * 0.05  # negative wh
            # logger.info(
            #     f"ui_sum: {ui_sum}",
            #     extra=dictLogger,
            # )
            logd.info(
                f"wh: {wh}",
                extra=dictLogger,
            )

            if (step_count % 2) == 0:  # only for even observation/reward take an action
                # k_cycle = 1000  # TODO determine the ratio
                # r_t += k_cycle * motion_magnitude.numpy()[0] # TODO add velocitoy sum as reward
                # wh0 = wh  # add velocitoy sum as reward
                prev_r_t = (wh1 + wh) * (
                    -1.0
                )  # most recent odd and even indexed reward
                episode_reward += prev_r_t
                # TODO add speed sum as positive reward

                if step_count > 0:
                    if step_count == 2:  # first even step has $r_0$
                        h_t = np.hstack([prev_o_t, prev_a_t, prev_r_t])
                    else:
                        h_t = np.append(
                            h_t, np.hstack([prev_o_t, prev_a_t, prev_r_t]), axis=0
                        )

                o_t0 = o_t

                o_t1 = tf.expand_dims(o_t0, 0)  # motion states is 30*3 matrix

                # predict action probabilities and estimated future rewards
                # from environment state
                # for causal rl, the odd indexed observation/reward are caused by last action
                # skip the odd indexed observation/reward for policy to make it causal
                logc.info(
                    f"E{epi_cnt} before inference!",
                    extra=dictLogger,
                )
                a_t = rdpg.actor_predict(o_t1, step_count / 2)

                prev_o_t = np.reshape(o_t0, [1, num_observations * obs_len])
                prev_a_t = np.reshape(a_t, [1, num_reduced_actions])

                logd.info(
                    f"E{epi_cnt} inference done with reduced action space!",
                    extra=dictLogger,
                )

                vcu_calib_table_reduced = tf.reshape(
                    a_t,
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
                pds_curr_table = pd.DataFrame(vcu_calib_table1, pd_index, pd_columns)
                # logc.info(
                #     f"E{epi_cnt} start record instant table: {step_count}",
                #     extra=dictLogger,
                # )

                if args.record_table:
                    curr_table_store_path = (
                        datafolder
                        + "/tables/instant_table_rdpg-"
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
                # r_t = (wh0 + wh) * (-1.0)  # odd + even indexed reward
                # Bugfix: the reward recorded in the first even step is not causal
                # cycle_reward should include the most recent even wh in the even step
                # record the odd step wh
                wh1 = wh

                # TODO add speed sum as positive reward
                logc.info(
                    f"E{epi_cnt} Step done: {step_count}",
                    extra=dictLogger,
                )

                # o_t = tf.stack([o_t0, o_t])
                # o_t_history was not used for back propagation
                # 60 frames, but never used again
                # o_t_history.append(o_t)

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

        rdpg.add_to_replay(h_t)
        if args.infer:
            (actor_loss, critic_loss) = rdpg.notrain()
            logd.info("No Learning, just calculating loss")
        else:
            for k in range(6):
                # logger.info(f"BP{k} starts.", extra=dictLogger)
                actor_loss, critic_loss = rdpg.train()
                logd.info("Learning and soft updating")
                # logd.info(f"BP{k} done.", extra=dictLogger)
                logd.info(
                    f"E{epi_cnt}BP{k} critic loss: {critic_loss}",
                    extra=dictLogger,
                )
                rdpg.soft_update_target()
                # logger.info(f"Updated target critic.", extra=dictLogger)

            # Checkpoint manager save model
            rdpg.save_ckpt()

        logd.info(
            f"E{epi_cnt} episode critic loss: {critic_loss}; episode actor loss: {actor_loss}.",
            extra=dictLogger,
        )
        # update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # generate 3d plot for ckpt
        fig = plot_3d_figure(vcu_calib_table1, pd_columns, pd_index)
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
        + "/last_table_rdpg-"
        + datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        + ".csv"
    )
    with open(last_table_store_path, "wb") as f:
        pds_last_table.to_csv(last_table_store_path)
    rdpg.save_replay_buffer()

    logc.info(f"main dies!!!!", extra=dictLogger)


if __name__ == "__main__":
    main()
