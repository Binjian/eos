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

# tracer = VizTracer()
# logging.basicConfig(level=logging.DEBUG, format=fmt)
logging.getLogger("matplotlib.font_manager").disabled = True

# logging.basicConfig(format=fmt)
logger = logging.getLogger("l045a")
formatter = logging.Formatter(
    "T(%(relativeCreated)d):Thr(%(threadName)s):F(%(funcName)s)L(%(lineno)d)C(%(user)s): Msg-%(message)s"
)
logfilename = (
    "../data/l045a_ac_tf_asyncio-"
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

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logger.info(f"Start Logging", extra=dictLogger)


import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

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

# asyncio module import
import asyncio

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
from comm.vcu_calib_generator import generate_vcu_calibration


# from communication import carla_ros
from agent import constructactorcriticnetwork_a2c, train_step_a2c
from comm import set_tbox_sim_path

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
train_log_dir = "logs/gradient_tape/" + current_time + "/train"
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

# # tableQueue contains a table which is a list of type float
# tableQueue = queue.Queue()
# # figQueue is for visualization thread to show the calibration 3d figure
# figQueue = queue.Queue()
# # motionpowerQueue contains a vcu states list with N(20) subsequent motion states + reward as observation
# motionpowerQueue = queue.Queue()
#
episode_done = False
episode_count = 0

states_rewards = []

wait_for_reset = True
vcu_step = False
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


pedal_range = [0, 1.0]
velocity_range = [0, 20.0]

# default table
vcu_calib_table0 = generate_vcu_calibration(
    vcu_calib_table_col, pedal_range, vcu_calib_table_row, velocity_range, 2
)
vcu_calib_table = np.copy(vcu_calib_table0)  # shallow copy of the default table
vcu_table = vcu_calib_table.reshape(-1).tolist()
# send_float_array("TQD_trqTrqSetNormal_MAP_v", vcu_table)
logger.info(f"flash initial table", extra=dictLogger)
# TQD_trqTrqSetECO_MAP_v

# # Create a matplotlib 3d figure, //export and save in log
# pd_data = pd.DataFrame(
#     vcu_calib_table,
#     columns=np.linspace(0, 1.0, num=17),
#     index=np.linspace(0, 30, num=21),
# )
# df = pd_data.unstack().reset_index()
# df.columns = ["pedal", "velocity", "throttle"]
#
# fig = plt.figure()
# ax = fig.gca(projection="3d")
# surf = ax.plot_trisurf(
#     df["pedal"],
#     df["velocity"],
#     df["throttle"],
#     cmap=plt.cm.viridis,
#     linewidth=0.2,
# )
# fig.colorbar(surf, shrink=0.5, aspect=5)
# ax.view_init(30, 135)
# plt.show()
# time.sleep(10)
# plt.close()

# vcu_act_list = vcu_calib_table.numpy().reshape(-1).tolist()
# create actor-critic network
num_observations = 2  # observed are the current speed and throttle; !! acceleration not available in l045a
sequence_len = 30  # 40 observation pairs as a valid observation for agent, for period of 50ms, this is equal to 2 second
num_inputs = num_observations * sequence_len  # 60 subsequent observations
num_actions = vcu_calib_table_size  # 17*21 = 357
num_hidden = 128
bias_mu = 0.0  # bias 0.0 yields mu=0.0 with linear activation function
bias_sigma = 0.55  # bias 0.55 yields sigma=1.0 with softplus activation function
# checkpoint_path = "./checkpoints/cp-{epoch:04d}.ckpt"
# checkpoint_path = "../tf_ckpts/checkpoint"
# checkpoint_dir = os.path.dirname(checkpoint_path)
tf.keras.backend.set_floatx("float64")
# TODO option fix sigma, just optimize mu
# TODO create testing scenes for gathering data
actorcritic_network = constructactorcriticnetwork_a2c(
    num_observations, sequence_len, num_actions, num_hidden, bias_mu, bias_sigma
)
gamma = 0.99  # discount factor for past rewards
opt = keras.optimizers.Adam(learning_rate=0.001)
# add checkpoints manager
checkpoint_dir = "../tf_ckpts"
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=actorcritic_network)
manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=10)
ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    # logger.info(f"Restored from {manager.latest_checkpoint}")
    logger.info(f"Restored from {manager.latest_checkpoint}", extra=dictLogger)
else:
    # logger.info("Initializing from scratch")
    logger.info(f"Initializing from scratch", extra=dictLogger)

# # get hmi status from udp message
# def get_hmi_status():
#     global wait_for_reset, episode_done
#     s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#     s.bind(get_hmi_status.myHost, get_hmi_status.myPort)
#     s.listen(5)
#
#     while True:
#         connection, address = s.accept()
#         time.sleep(0.1)
#         print("Server connected by", address)
#         while True:
#             data = connection.recv(1024)
#             while not data:
#                 time.sleep(0.1)
#                 break
#             else:
#                 status = data["status"]
#                 with hmi_lock:
#                     if status == "begin":
#                         wait_for_reset = False
#                         episode_done = False
#                     elif status == "end":
#                         wait_for_reset = True
#                         episode_done = True
#                     time.sleep(0.1)
#             connection.close()
#     s.close()
#
#
# get_hmi_status.myHost = "127.0.0.1"
# get_hmi_status.myPort = "8002"

# @eye
# tracer.start()
logger.info(f"Global Initialization done!", extra=dictLogger)


async def get_truck_status(motionpowerQueue: asyncio.Queue) -> None:
    global episode_done, wait_for_reset
    logger.info(f"Enter observation!", extra=dictLogger)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)  # set port reusable for enabling nc to monitor udp
    socket.socket.settimeout(s, None)
    s.bind((get_truck_status.myHost, get_truck_status.myPort))
    # s.listen(5)
    # datetime.datetime.now().strftime("%Y%b%d-%H%M%S")
    start_moment = time.time()
    th_exit = False
    last_moment = time.time()
    logger.info(f"Observation initialization Done!", extra=dictLogger)

    while not th_exit:
        candata, addr = s.recvfrom(2048)
        # logger.info('Data received!!!', extra=dictLogger)
        pop_data = json.loads(candata)
        if len(pop_data) != 1:
            logging.critical("udp sending multiple shots!")
            break
        for key, value in pop_data.items():
            if key == "status":
                # print(candata)
                with hmi_lock:
                    if value == "begin":
                        get_truck_status.start = True
                        logger.info("%s", "Capture will start!!!", extra=dictLogger)
                        wait_for_reset = False
                        episode_done = False
                    elif value == "end_valid":
                        get_truck_status.start = False
                        logger.info("%s", "Capture will stop!!!", extra=dictLogger)
                        wait_for_reset = True
                        episode_done = True
                    elif value == "end_invalid":
                        get_truck_status.start = False
                        logger.info("%s", "Capture will stop!!!", extra=dictLogger)
                        wait_for_reset = True
                        episode_done = True
                    elif value == "exit":
                        logger.info("%s", "Capture will exit!!!", extra=dictLogger)
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
                    acc_pedal = float(value["pedal"])
                    brake_pressure = float(value["brake_pressure"])
                    current = float(value["A"])
                    voltage = float(value["V"])
                    power = float(value["W"])
                    motion_power = [
                        velocity,
                        acc_pedal,
                        brake_pressure,
                        current,
                        voltage,
                    ]
                    # motion_power = [velocity, acc_pedal, current, voltage]
                    step_moment = time.time()
                    # step_dt_object = datetime.datetime.fromtimestamp(step_moment)
                    send_moment = float(timestamp) / 1e06 - 28800
                    # send_dt_object = datetime.datetime.fromtimestamp(send_moment)

                    # print(step_moment-send_moment)
                    # print(step_moment, step_dt_object)
                    # print(send_moment, send_dt_object)
                    # print(step_moment-send_moment)
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
                        await motionpowerQueue.put(get_truck_status.motionpower_states)
                        await asyncio.sleep(0.1)
                        # watch(motionpowerQueue.qsize())
                        logger.info(
                            f"Producer creates {motionpowerQueue.qsize()}",
                            extra=dictLogger,
                        )
                        get_truck_status.motionpower_states = []
            else:
                continue
        # time.sleep(0.1)
    logger.info(f"get_truck_status dies!!!", extra=dictLogger)

    s.close()


get_truck_status.motionpower_states = []
get_truck_status.myHost = "127.0.0.1"
get_truck_status.myPort = 8002
get_truck_status.start = False

# this is the calibration table consumer for flashing
async def flash_vcu(tableQueue: asyncio.Queue) -> None:
    flash_count = 0
    th_exit = False
    logger.info(f"Flash task entered!", extra=dictLogger)
    while not th_exit:
        # time.sleep(0.1)
        with hmi_lock:
            if episode_done:
                th_exit = True
                break
            # print("1 tableQueue size: {}".format(tableQueue.qsize()))
        table = await tableQueue.get()  # default block = True
        tableQueue.task_done()
        # print("2 tableQueue size: {}".format(tableQueue.qsize()))
        # print("flash vcu calib table!")
        # output_path = "file://../data/Calib_table_{}.out".format(flash_count)
        # if flash_count % 2 == 0:
        #     table = np.zeros(17*21).tolist()
        # else:
        #     table = np.ones(17*21).tolist()

        # tf.print('calib table:', table, output_stream=output_path)
        # send_float_array("TQD_trqTrqSetNormal_MAP_v", table)
        flash_count += 1
        # time.sleep(1.0)
        await asyncio.sleep(1.0)
        logger.info(f"flash count:{flash_count}", extra=dictLogger)
        # watch(flash_count)

    logger.info(f"flash_vcu dies!!!", extra=dictLogger)


# # this is the figure consumer for visualization
# def show_calib_table(figqueue):
#     global episode_done, episode_count
#
#     while True:
#         time.sleep(0.1)
#         try:
#             figure = figqueue.get()  # default block = True
#         except queue.Empty:
#             pass
#         else:
#             plt.show(figure)
#             time.sleep(1)
#             plt.close(figure)
#             # print("show the calib table!")
#             # send_float_array('TQD_trqTrqSetECO_MAP_v', table)
#


async def learn(motionpowerQueue: asyncio.Queue, tableQueue: asyncio.Queue) -> None:
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
    episode_wh = 0
    motion_states_history = []

    logger.info(f"Learn Initialization done!", extra=dictLogger)
    while True:  # run until solved
        # if not close, wait for start signal from hmi
        with hmi_lock:
            done = episode_done
            if wait_for_reset:
                # time.sleep(0.1) # somehow this sleep is the reason why it took so long to start.
                # logger.info(f'wait for start!', extra=dictLogger)
                continue

        # hmi click start so episode start from here
        step_count = 0
        with tf.GradientTape() as tape:
            while not done:
                # TODO l045a define episode done (time, distance, defined end event)
                # obs, r, done, info = env.step(action)
                # episode_done = done
                motion_power = await motionpowerQueue.get()
                motionpowerQueue.task_done()

                logger.info(
                    f"Action start step {step_count}", extra=dictLogger
                )  # env.step(action) action is flash the vcu calibration table
                # watch(step_count)
                # reward history
                step = False
                motion_power_states = tf.convert_to_tensor(
                    motion_power
                )  # state must have 20 (speed, acceleration, throttle, current, voltage) 5 tuple
                # motion_states, power_states = tf.split(motion_power_states, [3, 2], 1)
                motion_states, power_states = tf.split(motion_power_states, [2, 2], 1)

                # motion_magnitude = tf.reduce_sum(tf.math.abs(motion_states), 0)
                # rewards should be a 30x2 matrix after split, if add brake_pressure, should be x3
                # reward is sum of power (U*I)
                vcu_reward = tf.reduce_sum(
                    tf.reduce_prod(power_states, 1)
                )  # vcu_reward is a scalar
                wh = vcu_reward / 3600.0 * 0.05  # negative wh
                k_vcu_reward = 1000  # TODO determine the ratio
                # vcu_reward += k_vcu_reward * motion_magnitude.numpy()[0] # add velocitoy sum as reward
                vcu_reward = -1.0 * wh  # add velocitoy sum as reward
                # TODO add speed sum as positive reward
                vcu_rewards_history.append(vcu_reward)
                episode_reward += vcu_reward
                episode_wh += wh

                motion_states_history.append(motion_states)
                motion_states = tf.expand_dims(
                    motion_states, 0
                )  # motion states is 20*3 matrix

                # predict action probabilities and estimated future rewards
                # from environment state
                mu_sigma, critic_value = actorcritic_network(motion_states)

                vcu_critic_value_history.append(critic_value[0, 0])
                mu_sigma_history.append(mu_sigma)

                # sample action from action probability distribution
                nn_mu, nn_sigma = tf.unstack(mu_sigma)
                mvn = tfd.MultivariateNormalDiag(loc=nn_mu, scale_diag=nn_sigma)
                vcu_action = mvn.sample()  # 17*21 =  357 actions
                vcu_action_history.append(vcu_action)
                # Here the lookup table with contrained output is part of the environemnt,
                # clip is part of the environment to be learned
                # action is not constrained!
                vcu_calib_table = tf.reshape(
                    vcu_action, [vcu_calib_table_row, vcu_calib_table_col]
                )
                # get change budget : % of initial table
                vcu_calib_table = tf.math.multiply(
                    vcu_calib_table * vcu_calib_table_budget, vcu_calib_table0
                )
                # add changes to the default value
                vcu_calib_table_min = 0.8 * vcu_calib_table0
                vcu_calib_table_max = 1.0 * vcu_calib_table0

                vcu_calib_table = tf.clip_by_value(
                    vcu_calib_table + vcu_calib_table0,
                    clip_value_min=vcu_calib_table_min,
                    clip_value_max=vcu_calib_table_max,
                )

                vcu_act_list = vcu_calib_table.numpy().reshape(-1).tolist()
                # tf.print('calib table:', vcu_act_list, output_stream=sys.stderr)
                await tableQueue.put(vcu_act_list)
                logger.info(
                    f"Action Push table: {tableQueue.qsize()}", extra=dictLogger
                )
                step_count += 1

                # time.sleep(0.9)  # this is a problem to add artificial delay (inappropriate workaround)

                with hmi_lock:
                    done = episode_done
                # if not done:
                #     # throw figure to the visualization thread
                #     figQueue.put(fig)

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
            ax = fig.gca(projection="3d")
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
            for r in vcu_rewards_history[::-1]:
                discounted_sum = r + gamma * discounted_sum
                returns.insert(0, discounted_sum)

            # normalize
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + epsilon)

            returns = returns.tolist()

            # calculating loss values to update our network
            history = zip(
                vcu_action_history, mu_sigma_history, vcu_critic_value_history, returns
            )

            # back propagation
            (
                loss_all,
                act_losses_all,
                entropy_losses_all,
                critic_losses_all,
            ) = train_step_a2c(actorcritic_network, history, opt, tape)
            ckpt.step.assign_add(1)

        with train_summary_writer.as_default():
            tf.summary.scalar("KWH", episode_wh, step=episode_count)
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

        output_template = "Episode {}, Loss all: {}, Act loss: {}, Entropy loss: {}, Critic loss: {}, Episode Reward: {}, Wh: {}"
        print(
            output_template.format(
                episode_count + 1,
                loss_all,
                act_losses_all,
                entropy_losses_all,
                critic_losses_all,
                episode_reward,
                episode_wh,
            )
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
        print(f"Saved checkpoint for step {int(ckpt.step)}: {save_path}")
        print(f"loss {loss_all.numpy():.2f}".format(loss_all.numpy()))

        episode_count += 1
        episode_reward = 0
        episode_wh = 0
        if episode_count % 1 == 0:
            print("========================")
            print(f"running reward: {running_reward:.2f} at episode {episode_count}")
        # for thread in threads:
        #     thread.join()
        #     print(f"{thread.getName()} exits!")
        logger.info(f"main dies!!!!", extra=dictLogger)
        # thread.exit()

        # need hmi exit signal to exit task properly
        break

        # TODO terminate condition to be defined: reward > limit (percentage); time too long
        # if running_reward > 195:  # condition to consider the task solved
        #     print("solved at episode {}!".format(episode_count))
        #     break

    # thr_observe.join()
    # thr_update.join()
    # thr_flash.join()


epsilon = np.finfo(np.float64).eps.item()  # smallest number such that 1.0 + eps != 1.0

# TODO replace threads by asyncio task
# TODO mix threading and asyncio
# TODO add a thread for send_float_array
# TODO add printing calibration table
# TODO add initialize table to EP input
# @eye
async def main():
    global episode_done, episode_count, wait_for_reset
    global states_rewards
    global vcu_step
    # ros msgs for vcu communication
    # rospy.init_node("carla", anonymous=True)
    # rospy.Subscriber("/newrizon/vcu_input", VCU_Input, get_motionpower)
    # rospy.Subscriber("/newrizon/vcu_reward", VCU_Reward, get_reward)

    tskq_observation = asyncio.Queue()
    tskq_table = asyncio.Queue()

    # Start thread for flashing vcu, flash first
    # thread.start_new_thread(get_hmi_status, ())
    # thread.start_new_thread(get_truck_status, ())
    # observe = Thread(target=get_truck_status, args=())
    tsk_observe = asyncio.create_task(get_truck_status(tskq_observation))
    tsk_learn = asyncio.create_task(learn(tskq_observation, tskq_table))
    tsk_flash = asyncio.create_task(flash_vcu(tskq_table))

    await asyncio.gather(tsk_observe, tsk_learn, tsk_flash)
    # await tsk_observe
    # await tsk_observe.join()

    await tskq_observation.join()
    await tskq_table.join()

    # tsk_learn.cancel()
    # tsk_flash.cancel()

    """
    ## visualizations
    in early stages of training:
    ![imgur](https://i.imgur.com/5gcs5kh.gif)
    
    in later stages of training:
    ![imgur](https://i.imgur.com/5ziizud.gif)
    """


# tracer.stop()
# tracer.save()

if __name__ == "__main__":
    s = time.perf_counter()
    asyncio.run(main())
    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed: 0.2f} seconds.")
