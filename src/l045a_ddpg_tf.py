"""
Title: Advantage Actor Critic Method
Author: [Binjian Xin](https://www.newrizon.com)
Date created: 2021/02/12
Last modified: 2020/03/15
Description: Implement Advantage Actor Critic Method in Carla environment.
"""
import sys

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
from birdseye import eye

# from viztracer import VizTracer
from watchpoints import watch


# Logging Service Initialization
import logging
import inspect

# tracer = VizTracer()
# logging.basicConfig(level=logging.DEBUG, format=fmt)
mpl_logger = logging.getLogger("matplotlib.font_manager")
mpl_logger.disabled = True

# logging.basicConfig(format=fmt)
logger = logging.getLogger("l045a")
logger.propagate = False
formatter = logging.Formatter(
    "T(%(relativeCreated)d):Thr(%(threadName)s):F(%(funcName)s)L(%(lineno)d): Msg-%(message)s"
)
logfilename = (
    "../data/l045a_ddpg_tf-"
    + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S_%f")[:-3]
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


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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


tf.keras.backend.set_floatx("float64")
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

from agent.ddpg import (
    get_actor,
    get_critic,
    policy,
    Buffer,
    update_target,
    OUActionNoise,
)

from comm.tbox.scripts.tbox_sim import *

set_tbox_sim_path("/home/is/devel/newrizon/drl-carla-manual/src/comm/tbox")
# value = [99.0] * 21 * 17
# send_float_array('TQD_trqTrqSetECO_MAP_v', value)

# TODO add vehicle communication insterface
# TODO add model checkpoint episodically unique


# multithreading initialization
vcu_step_lock = Lock()
hmi_lock = Lock()

# TODO add visualization and logging
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = "logs/ddpg/gradient_tape/" + current_time + "/train"
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

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
vcu_calib_table_size = vcu_calib_table_row * vcu_calib_table_col
action_budget = 0.10  # interval that allows modifying the calibration table
action_lower = 0.8
action_upper = 1.0
action_bias = 0.0

pedal_range = [0, 1.0]
velocity_range = [0, 20.0]

# default table
vcu_calib_table0 = generate_vcu_calibration(
    vcu_calib_table_col, pedal_range, vcu_calib_table_row, velocity_range, 2
)
vcu_calib_table = np.copy(vcu_calib_table0)  # shallow copy of the default table
vcu_table = vcu_calib_table.reshape(-1).tolist()
logger.info(f"Start flash initial table", extra=dictLogger)
time.sleep(1.0)
# send_float_array("TQD_trqTrqSetNormal_MAP_v", vcu_table)
logger.info(f"Done flash initial table", extra=dictLogger)
# TQD_trqTrqSetECO_MAP_v

# create actor-critic network
num_observations = 3  # observed are the current speed, throttle, brake percentage; !! acceleration not available in l045a
sequence_len = 30  # 30 observation pairs as a valid observation for agent, for period of 50ms, this is equal to 2 second
num_inputs = num_observations * sequence_len  # 60 subsequent observations
num_actions = vcu_calib_table_size  # 17*21 = 357

# hyperparameters for DRL
num_hidden = 256
num_hidden0 = 16
num_hidden1 = 32


# Initialize networks
actor_model = get_actor(
    num_observations,
    num_actions,
    sequence_len,
    num_hidden,
    action_bias,
    action_budget,
    action_lower,
    action_upper,
)

critic_model = get_critic(
    num_observations, num_actions, sequence_len, num_hidden0, num_hidden1, num_hidden
)


# Initialize networks
target_actor = get_actor(
    num_observations,
    num_actions,
    sequence_len,
    num_hidden,
    action_bias,
    action_budget,
    action_upper,
    action_lower,
)

target_critic = get_critic(
    num_observations, num_actions, sequence_len, num_hidden0, num_hidden1, num_hidden
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
    num_actions,
    buffer_capacity=50000,
    batch_size=4,
    gamma=0.99
)


# add checkpoints manager
checkpoint_actor_dir = "../tf_ckpts/ddpg-actor"
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

checkpoint_critic_dir = "../tf_ckpts/ddpg-critic"
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


# checkpoint_tactor_dir = "../tf_ckpts/ddpg-tactor"
# ckpt_tactor = tf.train.Checkpoint(step=tf.Variable(1), optimizer=actor_optimizer, net=target_actor)
# manager_tactor = tf.train.CheckpointManager(ckpt_tactor, checkpoint_tactor_dir, max_to_keep=10)
# ckpt_tactor.restore(manager_tactor.latest_checkpoint)
# if manager_tactor.latest_checkpoint:
#     logger.info(f"Target Actor Restored from {manager_tactor.latest_checkpoint}", extra=dictLogger)
# else:
#     logger.info(f"Target Actor Initializing from scratch", extra=dictLogger)
#
# checkpoint_tcritic_dir = "../tf_ckpts/ddpg-tcritic"
# ckpt_tcritic = tf.train.Checkpoint(step=tf.Variable(1), optimizer=critic_optimizer, net=target_critic)
# manager_tcritic = tf.train.CheckpointManager(ckpt_tcritic, checkpoint_tcritic_dir, max_to_keep=10)
# ckpt_tcritic.restore(manager_tcritic.latest_checkpoint)
# if manager_tcritic.latest_checkpoint:
#     logger.info(f"Target Critic Restored from {manager_tcritic.latest_checkpoint}", extra=dictLogger)
# else:
#     logger.info(f"Target Critic Initializing from scratch", extra=dictLogger)

# Making the weights equal initially after checkpoints load
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())


# todo ignites manual loading of tensorflow library, to guarantee the real-time processing of first data in main thread
init_motionpower = np.random.rand(sequence_len, num_observations)
init_states = tf.convert_to_tensor(
    init_motionpower
)  # state must have 30 (speed, acceleration, throttle, current, voltage) 5 tuple
init_states = tf.expand_dims(init_states, 0)  # motion states is 30*2 matrix

# noise is a row vector of num_actions dimension
std_dev = 0.2
ou_noise = OUActionNoise(
    mean=np.zeros(num_actions), std_deviation=float(std_dev) * np.ones(num_actions)
)

action0 = policy(actor_model, init_states, action_lower, action_upper, ou_noise)
logger.info(f"manual load tf library by calling convert_to_tensor", extra=dictLogger)
ou_noise.reset()

# @eye
# tracer.start()
logger.info(f"Global Initialization done!", extra=dictLogger)


def get_truck_status():
    global episode_done, wait_for_reset, program_exit, motionpowerQueue
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
                    elif value == "end_valid":
                        get_truck_status.start = False
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
                    # acceleration in invalid for l0415
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
                    )  # obs_reward [speed, acc, pedal, current, voltage]
                    # logger.info(f'motionpower: {motion_power}', extra=dictLogger)
                    if len(get_truck_status.motionpower_states) >= sequence_len:
                        # print(f"motion_power num: {len(get_truck_status.motionpower_states)}")
                        motionpowerQueue.put(get_truck_status.motionpower_states)
                        # watch(motionpowerQueue.qsize())
                        logger.info(
                            f"Producer creates {motionpowerQueue.qsize()}",
                            extra=dictLogger,
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

    logger.info("Initialization Done!", extra=dictLogger)
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
            logger.info("flash starts", extra=dictLogger)
            # send_float_array("TQD_trqTrqSetNormal_MAP_v", table)
            time.sleep(1.0)
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
    vcu_rewards_history = []
    running_reward = 0
    episode_reward = 0
    episode_wh = 0
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
        prev_motion_states = None
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
                    logger.info(f"Experience Collection ends!", extra=dictLogger)
                    continue
                elif episode_end and (not done):
                    logger.info(
                        f"Experience Collection is interrupted!", extra=dictLogger
                    )
                    continue

                try:
                    logger.warning(f"Wait for an object!!!", extra=dictLogger)
                    motionpower = motionpowerQueue.get(block=True, timeout=1.55)
                except queue.Empty:
                    logger.warning(f"No data in the Queue!!!", extra=dictLogger)
                    continue

                logger.info(
                    f"Action start step {step_count}", extra=dictLogger
                )  # env.step(action) action is flash the vcu calibration table
                # watch(step_count)
                # reward history
                motionpower_states = tf.convert_to_tensor(
                    motionpower
                )  # state must have 30 (speed, acceleration, throttle, current, voltage) 5 tuple
                motion_states, power_states = tf.split(motionpower_states, [2, 2], 1)

                logger.info(f"tensor convert and split!", extra=dictLogger)
                # rewards should be a 20x2 matrix after split
                # reward is sum of power (U*I)
                wh = tf.reduce_sum(
                    tf.reduce_prod(power_states, 1)
                )  # vcu_reward is a scalar
                wh = wh / 3600.0 * 0.05  # negative wh

                if (
                    step_count % 2
                ) == 0:  # only for even observation/reward take an action
                    # k_vcu_reward = 1000  # TODO determine the ratio
                    # vcu_reward += k_vcu_reward * motion_magnitude.numpy()[0] # TODO add velocitoy sum as reward
                    vcu_reward0 = -1.0 * wh  # add velocitoy sum as reward
                    # TODO add speed sum as positive reward

                    if step_count != 0:
                        buffer.record(
                            (prev_motion_states, prev_action, vcu_reward, motion_states)
                        )
                        logger.info(f"BP starts.", extra=dictLogger)
                        buffer.learn()

                        update_target(target_actor.variables, actor_model.variables, tau)
                        update_target(target_critic.variables, critic_model.variables, tau)
                        logger.info(f"BP stops.", extra=dictLogger)

                    # motion_states_history.append(motion_states)
                    motion_states0 = motion_states

                    motion_states0 = tf.expand_dims(
                        motion_states0, 0
                    )  # motion states is 30*2 matrix

                    # predict action probabilities and estimated future rewards
                    # from environment state
                    # for causl rl, the odd indexed observation/reward are caused by last action
                    # skip the odd indexed observation/reward for policy to make it causal
                    logger.info(f"before inference!", extra=dictLogger)
                    # mu_sigma, critic_value = actorcritic_network(motion_states0)
                    action = policy(
                        actor_model,
                        motion_states0,
                        action_lower,
                        action_upper,
                        ou_noise
                    )
                    prev_motion_states = motion_states0
                    prev_action = action

                    logger.info(f"inference done!", extra=dictLogger)
                    # vcu_critic_value_history.append(critic_value[0, 0])
                    # mu_sigma_history.append(mu_sigma)
                    #
                    # # sample action from action probability distribution
                    # nn_mu, nn_sigma = tf.unstack(mu_sigma)
                    # mvn = tfd.MultivariateNormalDiag(loc=nn_mu, scale_diag=nn_sigma)
                    # vcu_action = mvn.sample()  # 17*21 =  357 actions
                    # logger.info(f"sampling done!", extra=dictLogger)
                    # vcu_action_history.append(vcu_action)
                    # Here the lookup table with constrained output is part of the environment,
                    # clip is part of the environment to be learned
                    # action is not constrained!
                    vcu_calib_table = tf.reshape(
                        action, [vcu_calib_table_row, vcu_calib_table_col]
                    )
                    # # get change budget : % of initial table
                    # vcu_calib_table = tf.math.multiply(
                    #     vcu_calib_table, vcu_calib_table0
                    # )
                    # # add changes to the default value
                    # vcu_calib_table_min = 0.8 * vcu_calib_table0
                    # vcu_calib_table_max = 1.0 * vcu_calib_table0

                    # vcu_calib_table = tf.clip_by_value(
                    #     vcu_calib_table + vcu_calib_table0,
                    #     clip_value_min=vcu_calib_table_min,
                    #     clip_value_max=vcu_calib_table_max,
                    # )

                    vcu_act_list = vcu_calib_table.numpy().reshape(-1).tolist()
                    # tf.print('calib table:', vcu_act_list, output_stream=sys.stderr)
                    tableQueue.put(vcu_act_list)
                    logger.info(
                        f"Action Push table: {tableQueue.qsize()}", extra=dictLogger
                    )
                    logger.info(f"Step : {step_count}", extra=dictLogger)

                # during odd steps, old action remains effective due to learn and flash delay
                # so just record the reward history
                # motion states (observation) are not used later for backpropagation
                else:
                    vcu_reward = vcu_reward0 + (-1.0) * wh  # odd + even indexed reward
                    # TODO add speed sum as positive reward
                    vcu_rewards_history.append(vcu_reward)
                    episode_reward += vcu_reward
                    episode_wh += wh
                    logger.info(f"Step : {step_count}", extra=dictLogger)

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
                    f"Episode interrupted, waits for next episode kicking off!",
                    extra=dictLogger,
                )
                continue  # otherwise assuming the history is valid and back propagate

            output_path = f"file://../data/Calib_table_{episode_count}.out"
            tf.print("calib table:", vcu_calib_table, output_stream=output_path)
            # Create a matplotlib 3d figure, //export and save in log
            pd_data = pd.DataFrame(
                vcu_calib_table.numpy(),
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
            for r in vcu_rewards_history[::-1]:
                discounted_sum = r + gamma * discounted_sum
                returns.insert(0, discounted_sum)

            # normalize
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            returns = returns.tolist()

            # # calculating loss values to update our network
            # history = zip(
            #     vcu_action_history, mu_sigma_history, vcu_critic_value_history, returns
            # )

            # logger.info(f"BP starts.", extra=dictLogger)
            # buffer.learn()
            # update_target(target_actor.variables, actor_model.variables, tau)
            # update_target(target_critic.variables, critic_model.variables, tau)

            # back propagation
            # (
            #     loss_all,
            #     act_losses_all,
            #     entropy_losses_all,
            #     critic_losses_all,
            # ) = train_step(actorcritic_network, history, opt, tape)
            # logger.info(f"BP ends.", extra=dictLogger)
            ckpt_actor.step.assign_add(1)
            ckpt_critic.step.assign_add(1)

        with train_summary_writer.as_default():
            tf.summary.scalar("KWH", episode_wh, step=episode_count)
            # tf.summary.scalar("loss_sum", loss_all, step=episode_count)
            # tf.summary.scalar("loss_act", act_losses_all, step=episode_count)
            # tf.summary.scalar("loss_entropy", entropy_losses_all, step=episode_count)
            # tf.summary.scalar("loss_critic", critic_losses_all, step=episode_count)
            tf.summary.scalar("reward", episode_reward, step=episode_count)
            tf.summary.scalar("running reward", running_reward, step=episode_count)
            tf.summary.image(
                "Calibration Table", plot_to_image(fig), step=episode_count
            )
            tf.summary.histogram(
                "Calibration Table Hist", vcu_act_list, step=episode_count
            )
        plt.close(fig)

        output_template = "Episode {}, Loss all: {}, Act loss: {}, Entropy loss: {}, Critic loss: {}, Episode Reward: {}, Wh: {}"
        print(
            output_template.format(
                episode_count + 1,
                episode_reward,
                episode_wh,
            )
        )

        # Checkpoint manager save model
        save_path_actor = manager_actor.save()
        print(f"Saved checkpoint for step {int(ckpt_actor.step)}: {save_path_actor}")
        save_path_critic = manager_critic.save()
        print(f"Saved checkpoint for step {int(ckpt_actor.step)}: {save_path_critic}")

        episode_count += 1
        episode_reward = 0
        episode_wh = 0
        if episode_count % 1 == 0:
            print("========================")
            print(f"running reward: {running_reward:.2f} at episode {episode_count}")

        logger.info(
            f"Episode done, waits for next episode kicking off!", extra=dictLogger
        )
        # TODO terminate condition to be defined: reward > limit (percentage); time too long
        # if running_reward > 195:  # condition to consider the task solved
        #     print("solved at episode {}!".format(episode_count))
        #     break

    thr_observe.join()
    thr_flash.join()

    # Save the weights
    actor_model.save_weights("../model/veos_actor.h5")
    critic_model.save_weights("../model/veos_critic.h5")
    target_actor.save_weights("../model/veos_target_actor.h5")
    target_critic.save_weights("../model/veos_target_critic.h5")


    logger.info(f"main dies!!!!", extra=dictLogger)


if __name__ == "__main__":
    main()
