"""
Title: realtime_train_infer_ddpg
Author: [Binjian Xin](https://www.newrizon.com)
Date created: 2022/05/10
Last modified: 2022/05/10
Description: Implement realtime DDPG algorithm for training and inference

## Introduction

This script shows an implementation of DDPG method on l045a truck real environment.

### Deep Deterministic Policy Gradient (DDPG)

### Gym-Carla env

An Ego Vehicle drives through a fixed track and collect loss (negative reward) defined
as energy consumption

### References

- [DDPG ](https://keras.io/examples/rl/ddpg_pendulum/)

"""

# system imports
import os
import argparse
import datetime

import socket
import json

from threading import Lock, Thread
import time, queue, math, signal

# third party imports
from collections import deque
import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
from tensorflow.python.client import device_lib

## visualization import
import pandas as pd
import matplotlib.pyplot as plt

## logging
import logging
from logging.handlers import SocketHandler
import inspect
from pythonjsonlogger import jsonlogger

# local imports

from .visualization import plot_to_image
from .comm import generate_vcu_calibration, set_tbox_sim_path, send_float_array
from .agent import get_actor, get_critic, policy, Buffer, update_target, OUActionNoise
from . import logger, dictLogger, projroot

# from utils import get_logger, get_truck_status, flash_vcu, plot_3d_figure
# set_tbox_sim_path("/home/veos/devel/newrizon/drl-carla-manual/src/comm/tbox")
# value = [99.0] * 21 * 17
# send_float_array('TQD_trqTrqSetECO_MAP_v', value)

# global variables: threading, data, lock, etc.
class realtime_train_infer_ddpg(object):
    def __init__(
        self,
        resume=False,
        infer=False,
        record=True,
        path=".",
        projroot=".",
        logger=None,
    ):
        self.projroot = projroot
        self.logger = logger
        self.dictLogger = dictLogger
        # self.dictLogger = {"user": inspect.currentframe().f_code.co_name}
        self.resume = resume
        self.infer = infer
        self.record = record
        self.path = path

        self.eps = np.finfo(np.float32).eps.item()  # smallest number such that 1.0 + eps != 1.0

        if resume:
            self.dataroot = projroot.joinpath("data/" + self.path)
        else:
            self.dataroot = projroot.joinpath("data/scratch/" + self.path)

        self.set_logger()
        self.logger.info(f"Start Logging", extra=self.dictLogger)

        self.set_data_path()
        tf.keras.backend.set_floatx("float32")
        self.logger.info(
            f"tensorflow device lib:\n{device_lib.list_local_devices()}\n",
            extra=self.dictLogger,
        )
        self.logger.info(f"Tensorflow Imported!", extra=self.dictLogger)

        self.init_vehicle()
        self.build_actor_critic()
        self.init_checkpoint()
        self.flash_vcu_once()
        self.logger.info(f"VCU and GPU Initialization done!", extra=self.dictLogger)
        self.init_threads_data()
        self.logger.info(f"Thread data Initialization done!", extra=self.dictLogger)

    def set_logger(self):
        self.logroot = self.dataroot.joinpath("py_logs")
        try:
            os.makedirs(self.logroot)
        except FileExistsError:
            print("User folder exists, just resume!")

        logfilename = self.logroot.joinpath(
            "eos-rt-ddpg"
            + datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
            + ".log"
        )
        formatter = logging.Formatter(
            "%(asctime)s-%(name)s-%(levelname)s-%(module)s-%(threadName)s-%(funcName)s)-%(lineno)d): %(message)s"
        )
        json_file_formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(module)s %(threadName)s %(funcName)s) %(lineno)d) %(message)s"
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

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        self.logger.addHandler(sh)

        self.logger.setLevel(logging.DEBUG)
        # self.dictLogger = {'funcName': '__self__.__func__.__name__'}
        # self.dictLogger = {'user': inspect.currentframe().f_back.f_code.co_name}

        self.logc = logger.getChild("control flow")
        self.logc.propagate = True
        self.logd = logger.getChild("data flow")
        self.logd.propagate = True

    def set_data_path(self):
        # Create folder for ckpts loggings.
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = self.dataroot.joinpath(
            "tf_logs/ddpg/gradient_tape/" + current_time + "/train"
        )
        self.train_summary_writer = tf.summary.create_file_writer(str(self.train_log_dir))
        # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        if self.resume:
            self.logger.info(f"Resume last training", extra=self.self.dictLogger)
        else:
            self.logger.info(f"Start from scratch", extra=self.dictLogger)

    def init_vehicle(self):
        # resume last pedal map / scratch from default table
        set_tbox_sim_path(str(self.projroot.joinpath("eos/comm/tbox")))

        # initialize pedal map parameters
        self.vcu_calib_table_col = 17  # number of pedal steps, x direction
        self.vcu_calib_table_row = 21  # numnber of velocity steps, y direction
        self.vcu_calib_table_budget = (
            0.05  # interval that allows modifying the calibration table
        )
        self.vcu_calib_table_size = self.vcu_calib_table_row * self.vcu_calib_table_col
        self.action_budget = (
            0.10  # interval that allows modifying the calibration table
        )
        self.action_lower = 0.8
        self.action_upper = 1.0
        self.action_bias = 0.0

        self.pd_index = np.linspace(0, 100, self.vcu_calib_table_row)
        self.pd_index[1] = 7
        self.pd_columns = (
            np.array([0, 2, 4, 8, 12, 16, 20, 24, 28, 32, 38, 44, 50, 62, 74, 86, 100])
            / 100
        )

        self.target_velocity = np.array(
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

        self.pedal_range = [0, 1.0]
        self.velocity_range = [0, 20.0]

        if self.resume:
            self.vcu_calib_table0 = generate_vcu_calibration(
                self.vcu_calib_table_col,
                self.pedal_range,
                self.vcu_calib_table_row,
                self.velocity_range,
                3,
                self.dataroot,
            )
        else:
            self.vcu_calib_table0 = generate_vcu_calibration(
                self.vcu_calib_table_col,
                self.pedal_range,
                self.vcu_calib_table_row,
                self.velocity_range,
                2,
                self.dataroot,
            )
        self.vcu_calib_table1 = np.copy(
            self.vcu_calib_table0
        )  # shallow copy of the default table
        vcu_table1 = self.vcu_calib_table1.reshape(-1).tolist()
        self.logger.info(f"Start flash initial table", extra=self.dictLogger)
        # time.sleep(1.0)
        send_float_array("TQD_trqTrqSetNormal_MAP_v", vcu_table1, sw_diff=False)
        self.logger.info(f"Done flash initial table", extra=self.dictLogger)

        # TQD_trqTrqSetECO_MAP_v

    def build_actor_critic(self):
        """Builds the actor-critic network.

        this network learns two functions:
        1. actor: this takes as input the state of our environment and returns a
        probability value for each action in its action space.
        2. critic: this takes as input the state of our environment and returns
        an estimate of total rewards in the future.

        in our implementation, they share the initial layer.

        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """

        # create actor-critic network
        self.num_observations = 3  # observed are velocity, throttle, brake percentage; !! acceleration not available in l045a
        self.sequence_len = 30  # 30 observation pairs as a valid observation for agent, for period of 50ms, this is equal to 1.5 second
        self.num_inputs = (
            self.num_observations * self.sequence_len
        )  # 60 subsequent observations
        self.num_actions = self.vcu_calib_table_size  # 17*21 = 357
        self.vcu_calib_table_row_reduced = 5  # 0:5 adaptive rows correspond to low speed from  0~20, 7~25, 10~30, 15~35, etc  kmh  # overall action space is the whole table
        self.num_reduced_actions = (
            self.vcu_calib_table_row_reduced * self.vcu_calib_table_col
        )  # 5x17=85
        # hyperparameters for DRL
        self.num_hidden = 256
        self.num_hidden0 = 16
        self.num_hidden1 = 32

        # DYNAMIC: need to adapt the pointer to change different roi of the pm, change the starting row index
        self.vcu_calib_table_row_start = 0
        self.vcu_calib_table0_reduced = self.vcu_calib_table0[
            self.vcu_calib_table_row_start : self.vcu_calib_table_row_reduced
            + self.vcu_calib_table_row_start,
            :,
        ]

        # Initialize networks
        self.actor_model = get_actor(
            self.num_observations,
            self.num_reduced_actions,
            self.sequence_len,
            self.num_hidden,
            self.action_bias,
        )

        self.critic_model = get_critic(
            self.num_observations,
            self.num_reduced_actions,
            self.sequence_len,
            self.num_hidden0,
            self.num_hidden1,
            self.num_hidden,
        )

        # Initialize networks
        self.target_actor = get_actor(
            self.num_observations,
            self.num_reduced_actions,
            self.sequence_len,
            self.num_hidden,
            self.action_bias,
        )

        self.target_critic = get_critic(
            self.num_observations,
            self.num_reduced_actions,
            self.sequence_len,
            self.num_hidden0,
            self.num_hidden1,
            self.num_hidden,
        )

        # Learning rate for actor-critic models
        self.critic_lr = 0.002
        self.actor_lr = 0.001

        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)

        # Discount factor for future rewards
        self.gamma = 0.99
        # Used to update target networks
        self.tau = 0.005
        self.batch_size = 4
        self.buffer_capacity = 300000
        # try buffer size with 1,000,000

        self.buffer = Buffer(
            self.actor_model,
            self.critic_model,
            self.target_actor,
            self.target_critic,
            self.actor_optimizer,
            self.critic_optimizer,
            self.num_observations,
            self.sequence_len,
            self.num_reduced_actions,
            buffer_capacity=self.buffer_capacity,
            batch_size=self.batch_size,
            gamma=self.gamma,
            datafolder=str(self.dataroot),
        )

        # ou_noise is a row vector of num_actions dimension
        self.ou_noise_std_dev = 0.2
        self.ou_noise = OUActionNoise(
            mean=np.zeros(self.num_reduced_actions),
            std_deviation=float(self.ou_noise_std_dev) * np.ones(self.num_reduced_actions),
        )

    def init_checkpoint(self):
        # add checkpoints manager
        if self.resume:
            checkpoint_actor_dir = self.dataroot.joinpath("tf_ckpts-aa/l045a_ddpg_actor")
            checkpoint_critic_dir = self.dataroot.joinpath("tf_ckpts-aa/l045a_ddpg_critic")
        else:
            checkpoint_actor_dir = self.dataroot.joinpath(
                 "tf_ckpts-aa/l045a_ddpg_actor"
                + datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
            )
            checkpoint_critic_dir = self.dataroot.joinpath(
                "tf_ckpts-aa/l045a_ddpg_critic"
                + datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
            )
        try:
            os.makedirs(checkpoint_actor_dir)
            self.logger.info("Actor folder doesn't exist. Created!", extra=self.dictLogger)
        except FileExistsError:
            self.logger.info("Actor folder exists, just resume!", extra=self.dictLogger)
        try:
            os.makedirs(checkpoint_critic_dir)
            self.logger.info("User folder doesn't exist. Created!", extra=self.dictLogger)
        except FileExistsError:
            self.logger.info("User folder exists, just resume!", extra=self.dictLogger)

        self.ckpt_actor = tf.train.Checkpoint(
            step=tf.Variable(1), optimizer=self.actor_optimizer, net=self.actor_model
        )
        self.manager_actor = tf.train.CheckpointManager(
            self.ckpt_actor, checkpoint_actor_dir, max_to_keep=10
        )
        self.ckpt_actor.restore(self.manager_actor.latest_checkpoint)
        if self.manager_actor.latest_checkpoint:
            self.logger.info(
                f"Actor Restored from {self.manager_actor.latest_checkpoint}",
                extra=self.dictLogger,
            )
        else:
            self.logger.info(f"Actor Initializing from scratch", extra=self.dictLogger)

        self.ckpt_critic = tf.train.Checkpoint(
            step=tf.Variable(1), optimizer=self.critic_optimizer, net=self.critic_model
        )
        self.manager_critic = tf.train.CheckpointManager(
            self.ckpt_critic, checkpoint_critic_dir, max_to_keep=10
        )
        self.ckpt_critic.restore(self.manager_critic.latest_checkpoint)
        if self.manager_critic.latest_checkpoint:
            self.logger.info(
                f"Critic Restored from {self.manager_critic.latest_checkpoint}",
                extra=self.dictLogger,
            )
        else:
            self.logger.info("Critic Initializing from scratch", extra=self.dictLogger)

        # Making the weights equal initially after checkpoints load
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

    def flash_vcu_once(self):
        # ignites manual loading of tensorflow library, to guarantee the real-time processing of first data in main thread
        init_motionpower = np.random.rand(self.sequence_len, self.num_observations)
        init_states = tf.convert_to_tensor(
            init_motionpower
        )  # state must have 30 (speed, throttle, current, voltage) 5 tuple
        init_states = tf.expand_dims(init_states, 0)  # motion states is 30*2 matrix

        action0 = policy(self.actor_model, init_states, self.ou_noise)
        self.logger.info(
            f"manual load tf library by calling convert_to_tensor", extra=self.dictLogger
        )
        self.ou_noise.reset()

    # @eye
    # tracer.start()

    def reset_capture_handler(self):
        self.get_truck_status_start = False
        self.logger.info(f"reset_capture_handler called", extra=self.dictLogger)
        raise Exception("reset capture to stop")

    def init_threads_data(self):
        # multithreading initialization
        self.hmi_lock = Lock()

        # tableQueue contains a table which is a list of type float
        self.tableQueue = queue.Queue()
        # motionpowerQueue contains a vcu states list with N(20) subsequent motion states + reward as observation
        self.motionpowerQueue = queue.Queue()

        # initial status of the switches
        self.program_exit = False
        self.episode_done = False
        self.episode_end = False
        self.episode_count = 0

        signal.signal(signal.SIGALRM, self.reset_capture_handler)
        self.get_truck_status_start = False
        self.get_truck_status_motpow_t = []
        self.get_truck_status_myHost = "127.0.0.1"
        self.get_truck_status_myPort = 8002
        self.get_truck_status_start = False
        self.get_truck_status_qobject_len = 12  # sequence length 1.5*12s

    def get_truck_status(self):
        # global program_exit
        # global motionpowerQueue, sequence_len
        # global episode_count, episode_done, episode_end
        # global vcu_calib_table_row_start

        # self.logger.info(f'Start Initialization!', extra=self.dictLogger)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        socket.socket.settimeout(s, None)
        s.bind((self.get_truck_status_myHost, self.get_truck_status_myPort))
        # s.listen(5)
        # datetime.datetime.now().strftime("%Y%b%d-%H%M%S")
        start_moment = time.time()
        th_exit = False
        last_moment = time.time()
        self.logc.info(f"Initialization Done!", extra=self.dictLogger)
        # qobject_size = 0

        vel_hist_dQ = deque(maxlen=20)  # accumulate 1s of velocity values
        # vel_cycle_dQ = deque(maxlen=30)  # accumulate 1.5s (one cycle) of velocity values
        vel_cycle_dQ = deque(
            maxlen=self.sequence_len
        )  # accumulate 1.5s (one cycle) of velocity values

        while not th_exit:  # th_exit is local; program_exit is global
            with self.hmi_lock:  # wait for tester to kick off or to exit
                if self.program_exit == True:  # if program_exit is True, exit thread
                    self.logger.info(
                        "%s",
                        "Capture thread exit due to processing request!!!",
                        extra=self.dictLogger,
                    )
                    th_exit = True
                    continue
            candata, addr = s.recvfrom(2048)
            # self.logger.info('Data received!!!', extra=self.dictLogger)
            pop_data = json.loads(candata)
            data_type = type(pop_data)
            self.logc.info(f"Data type is {data_type}", extra=self.dictLogger)
            if not isinstance(pop_data, dict):
                self.logd.critical(f"udp sending wrong data type!", extra=self.dictLogger)
                raise TypeError("udp sending wrong data type!")

            epi_delay_stop = False
            for key, value in pop_data.items():
                if key == "status":  # state machine chores
                    # print(candata)
                    if value == "begin":
                        self.get_truck_status_start = True
                        self.logc.info("%s", "Episode will start!!!", extra=self.dictLogger)
                        th_exit = False
                        # ts_epi_start = time.time()

                        vel_hist_dQ.clear()
                        epi_delay_stop = False
                        with self.hmi_lock:
                            self.episode_done = False
                            self.episode_end = False

                    elif value == "end_valid":
                        # DONE for valid end wait for another 2 queue objects (3 seconds) to get the last reward!
                        # cannot sleep the thread since data capturing in the same thread, use signal alarm instead
                        self.get_truck_status_start = (
                            True  # do not stopping data capture immediately
                        )
                        self.get_truck_status_motpow_t = []
                        while not self.motionpowerQueue.empty():
                            self.motionpowerQueue.get()
                        self.logc.info("%s", "Episode done!!!", extra=self.dictLogger)
                        th_exit = False
                        vel_hist_dQ.clear()
                        epi_delay_stop = True
                        with self.hmi_lock:
                            self.episode_count += 1  # valid round increments
                            self.episode_done = True
                            self.episode_end = True
                    elif value == "end_invalid":
                        self.get_truck_status_start = False
                        self.logc.info(f"Episode is interrupted!!!", extra=self.dictLogger)
                        self.get_truck_status_motpow_t = []
                        vel_hist_dQ.clear()
                        # motionpowerQueue.queue.clear()
                        # self.logc.info(
                        #     f"Episode motionpowerQueue has {motionpowerQueue.qsize()} states remaining",
                        #     extra=self.dictLogger,
                        # )
                        while not self.motionpowerQueue.empty():
                            self.motionpowerQueue.get()
                        # self.logc.info(
                        #     f"Episode motionpowerQueue gets cleared!", extra=self.dictLogger
                        # )
                        th_exit = False
                        epi_delay_stop = False
                        with self.hmi_lock:
                            self.episode_done = False
                            self.episode_end = True
                            self.episode_count += 1  # invalid round increments
                    elif value == "exit":
                        self.get_truck_status_start = False
                        self.get_truck_status_motpow_t = []
                        vel_hist_dQ.clear()
                        while not self.motionpowerQueue.empty():
                            self.motionpowerQueue.get()
                        # self.logc.info("%s", "Program will exit!!!", extra=self.dictLogger)
                        th_exit = True
                        epi_delay_stop = False
                        # for program exit, need to set episode states
                        # final change to inform main thread
                        with self.hmi_lock:
                            self.episode_done = False
                            self.episode_end = True
                            self.program_exit = True
                            self.episode_count += 1
                        break
                        # time.sleep(0.1)
                elif key == "data":
                    # self.logger.info('Data received before Capture starting!!!', extra=self.dictLogger)
                    # self.logger.info(f'ts:{value["timestamp"]}vel:{value["velocity"]}ped:{value["pedal"]}', extra=self.dictLogger)
                    # DONE add logic for episode valid and invalid
                    if epi_delay_stop:
                        signal.alarm(3)  # delay stop for 3 seconds
                    try:
                        if self.get_truck_status_start:  # starts episode

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

                            self.get_truck_status_motpow_t.append(
                                motion_power
                            )  # obs_reward [speed, pedal, brake, current, voltage]
                            vel_hist_dQ.append(velocity)
                            vel_cycle_dQ.append(velocity)

                            if len(self.get_truck_status_motpow_t) >= self.sequence_len:
                                if len(vel_cycle_dQ) != vel_cycle_dQ.maxlen:
                                    self.logc.warning(  # the recent 1.5s average velocity
                                        f"cycle deque is inconsistent!",
                                        extra=self.dictLogger,
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
                                    self.vcu_calib_table_row_start = 0
                                elif vel_max < 100:
                                    self.vcu_calib_table_row_start = (
                                        math.floor((vel_max - 20) / 5) + 1
                                    )
                                else:
                                    self.logc.warning(
                                        f"cycle higher than 100km/h!",
                                        extra=self.dictLogger,
                                    )
                                    self.vcu_calib_table_row_start = 16

                                self.logd.info(
                                    f"Cycle velocity: Aver{vel_aver},Min{vel_min},Max{vel_max},StartIndex{self.vcu_calib_table_row_start}!",
                                    extra=self.dictLogger,
                                )
                                # self.logd.info(
                                #     f"Producer Queue has {motionpowerQueue.qsize()}!",
                                #     extra=self.dictLogger,
                                # )
                                self.motionpowerQueue.put(self.get_truck_status_motpow_t)
                                self.get_truck_status_motpow_t = []
                    except Exception as X:
                        self.logc.info(
                            X,  # f"Valid episode, Reset data capturing to stop after 3 seconds!",
                            extra=self.dictLogger,
                        )
                else:
                    self.logc.critical(
                        "udp sending unknown signal (neither status nor data)!"
                    )
                    break

        self.logger.info(f"get_truck_status dies!!!", extra=self.dictLogger)

        s.close()

    # this is the calibration table consumer for flashing
    # @eye
    def flash_vcu(self):

        flash_count = 0
        th_exit = False

        self.logc.info(f"Initialization Done!", extra=self.dictLogger)
        while not th_exit:
            # time.sleep(0.1)
            with self.hmi_lock:
                if self.program_exit:
                    th_exit = True
                    continue
            try:
                # print("1 tablequeue size: {}".format(tablequeue.qsize()))
                table = self.tableQueue.get(block=False, timeout=1)  # default block = True
                # print("2 tablequeue size: {}".format(tablequeue.qsize()))
            except queue.Empty:
                pass
            else:

                # tf.print('calib table:', table, output_stream=output_path)
                self.logc.info(f"flash starts", extra=self.dictLogger)
                send_float_array("TQD_trqTrqSetNormal_MAP_v", table, sw_diff=True)
                # time.sleep(1.0)
                self.logc.info(f"flash done, count:{flash_count}", extra=self.dictLogger)
                flash_count += 1
                # watch(flash_count)

        # motionpowerQueue.join()
        self.logc.info(f"flash_vcu dies!!!", extra=self.dictLogger)


    # @eye
    def run(self):
        # global episode_count
        # global program_exit
        # global motionpowerQueue
        # global pd_index, pd_columns
        # global episode_done, episode_end
        # global vcu_calib_table_row_start


        # Start thread for flashing vcu, flash first
        thr_observe = Thread(target=self.get_truck_status, name="observe", args=())
        thr_flash = Thread(target=self.flash_vcu, name="flash", args=())
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

        self.logger.info(f"main Initialization done!", extra=self.dictLogger)
        while not th_exit:  # run until solved or program exit; th_exit is local
            with self.hmi_lock:  # wait for tester to kick off or to exit
                th_exit = self.program_exit  # if program_exit is False,
                epi_cnt = self.episode_count  # get episode counts
                epi_end = self.episode_end
            if epi_end:  # if episode_end is True, wait for start of episode
                # self.logger.info(f'wait for start!', extra=self.dictLogger)
                continue

            step_count = 0
            wh1 = 0  # initialize odd step wh
            tf.summary.trace_on(graph=True, profiler=True)

            self.logc.info("----------------------", extra=self.dictLogger)
            self.logc.info(
                f"E{epi_cnt} starts!",
                extra=self.dictLogger,
            )
            with tf.GradientTape() as tape:
                while (
                    not epi_end
                ):  # end signal, either the round ends normally or user interrupt
                    # TODO l045a define round done (time, distance, defined end event)
                    with self.hmi_lock:  # wait for tester to interrupt or to exit
                        th_exit = self.program_exit  # if program_exit is False, reset to wait
                        epi_end = self.episode_end
                        done = self.episode_done
                        table_start = self.vcu_calib_table_row_start

                    if epi_end:  # stop observing and inferring
                        continue

                    try:
                        self.logc.info(
                            f"E{epi_cnt} Wait for an object!!!", extra=self.dictLogger
                        )
                        motionpower = self.motionpowerQueue.get(block=True, timeout=1.55)
                    except queue.Empty:
                        self.logc.info(
                            f"E{epi_cnt} No data in the Queue!!!",
                            extra=self.dictLogger,
                        )
                        continue

                    self.logc.info(
                        f"E{epi_cnt} start step {step_count}",
                        extra=self.dictLogger,
                    )  # env.step(action) action is flash the vcu calibration table
                    # watch(step_count)
                    # reward history
                    motionpower_states = tf.convert_to_tensor(
                        motionpower
                    )  # state must have 30 (velocity, pedal, brake, current, voltage) 5 tuple (num_observations)
                    motion_states, power_states = tf.split(motionpower_states, [3, 2], 1)

                    self.logd.info(
                        f"E{epi_cnt} tensor convert and split!",
                        extra=self.dictLogger,
                    )
                    ui_sum = tf.reduce_sum(
                        tf.reduce_prod(power_states, 1)
                    )  # vcu reward is a scalar
                    wh = ui_sum / 3600.0 * 0.05  # negative wh
                    # self.logger.info(
                    #     f"ui_sum: {ui_sum}",
                    #     extra=self.dictLogger,
                    # )
                    self.logd.info(
                        f"wh: {wh}",
                        extra=self.dictLogger,
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
                            self.buffer.record(
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
                        self.logc.info(
                            f"E{epi_cnt} before inference!",
                            extra=self.dictLogger,
                        )
                        vcu_action_reduced = policy(self.actor_model, motion_states1, self.ou_noise)
                        prev_motion_states = motion_states0
                        prev_action = vcu_action_reduced

                        self.logd.info(
                            f"E{epi_cnt} inference done with reduced action space!",
                            extra=self.dictLogger,
                        )

                        vcu_calib_table_reduced = tf.reshape(
                            vcu_action_reduced,
                            [self.vcu_calib_table_row_reduced, self.vcu_calib_table_col],
                        )
                        # self.logger.info(
                        #     f"vcu action table reduced generated!", extra=self.dictLogger
                        # )
                        # vcu_action_table_reduced_s = [f"{col:.3f},"
                        #                               for row in vcu_calib_table_reduced
                        #                               for col in row]
                        # self.logger.info(
                        #     f"vcu action table: {vcu_action_table_reduced_s}",
                        #     extra=self.dictLogger,
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
                        vcu_calib_table0_reduced = self.vcu_calib_table0[
                            table_start : self.vcu_calib_table_row_reduced + table_start,
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
                        self.vcu_calib_table1[
                            table_start : self.vcu_calib_table_row_reduced + table_start, :
                        ] = vcu_calib_table_reduced.numpy()
                        pds_curr_table = pd.DataFrame(
                            self.vcu_calib_table1, self.pd_index, self.pd_columns
                        )
                        # self.logc.info(
                        #     f"E{epi_cnt} start record instant table: {step_count}",
                        #     extra=self.dictLogger,
                        # )

                        if args.record_table:
                            curr_table_store_path = self.dataroot.joinpath(
                                "tables/instant_table_ddpg-bigep"
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
                            self.logd.info(
                                f"E{epi_cnt} done with record instant table: {step_count}",
                                extra=self.dictLogger,
                            )

                        vcu_act_list = self.vcu_calib_table1.reshape(-1).tolist()
                        # tf.print('calib table:', vcu_act_list, output_stream=sys.stderr)
                        self.tableQueue.put(vcu_act_list)
                        self.logd.info(
                            f"E{epi_cnt}StartIndex{table_start} Action Push table: {self.tableQueue.qsize()}",
                            extra=self.dictLogger,
                        )
                        self.logc.info(
                            f"E{epi_cnt} Finish Step: {step_count}",
                            extra=self.dictLogger,
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
                        self.logc.info(
                            f"E{epi_cnt} Step done: {step_count}",
                            extra=self.dictLogger,
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
                    self.logc.info(
                        f"E{epi_cnt} interrupted, waits for next episode to kick off!",
                        extra=self.dictLogger,
                    )
                    episode_reward = 0.0
                    continue  # otherwise assuming the history is valid and back propagate

                self.logc.info(
                    f"E{epi_cnt} Experience Collection ends!",
                    extra=self.dictLogger,
                )

                if self.infer:
                    (critic_loss, actor_loss) = self.buffer.nolearn()
                    self.logd.info("No Learning, just calculating loss")
                else:
                    for k in range(6):
                        # self.logger.info(f"BP{k} starts.", extra=self.dictLogger)
                        (critic_loss, actor_loss) = self.buffer.learn()
                        self.logd.info("Learning and updating")

                        update_target(self.target_actor.variables, self.actor_model.variables, self.tau)
                        # self.logger.info(f"Updated target actor", extra=self.dictLogger)
                        update_target(self.target_critic.variables, self.critic_model.variables, self.tau)
                        # self.logger.info(f"Updated target critic.", extra=self.dictLogger)

                    # Checkpoint manager save model
                    self.ckpt_actor.step.assign_add(1)
                    self.ckpt_critic.step.assign_add(1)
                    if int(self.ckpt_actor.step) % 5 == 0:
                        save_path_actor = self.manager_actor.save()
                        self.logd.info(
                            f"Saved checkpoint for step {int(self.ckpt_actor.step)}: {save_path_actor}",
                            extra=self.dictLogger,
                        )
                    if int(self.ckpt_critic.step) % 5 == 0:
                        save_path_critic = self.manager_critic.save()
                        self.logd.info(
                            f"Saved checkpoint for step {int(self.ckpt_actor.step)}: {save_path_critic}",
                            extra=self.dictLogger,
                        )

                # self.logd.info(f"BP{k} done.", extra=self.dictLogger)
                self.logd.info(
                    f"E{epi_cnt}BP{k} critic loss: {critic_loss}; actor loss: {actor_loss}",
                    extra=self.dictLogger,
                )

                # Create a matplotlib 3d figure, //export and save in log
                pd_data = pd.DataFrame(
                    self.vcu_calib_table1,
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
            # use local episode counter epi_cnt_local tf.summary.writer; otherwise specify multiple self.logdir and automatic switch
            with self.train_summary_writer.as_default():
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

            self.logd.info(
                f"E{epi_cnt} Episode Reward: {episode_reward}",
                extra=self.dictLogger,
            )

            episode_reward = 0
            self.logc.info(
                f"E{epi_cnt} done, waits for next episode to kick off!",
                extra=self.dictLogger,
            )
            self.logc.info("----------------------", extra=self.dictLogger)
            if epi_cnt % 10 == 0:
                self.logc.info("++++++++++++++++++++++++", extra=self.dictLogger)
                self.logc.info(
                    f"Running reward: {running_reward:.2f} at E{epi_cnt}",
                    extra=self.dictLogger,
                )
                self.logc.info("++++++++++++++++++++++++", extra=self.dictLogger)

            # TODO terminate condition to be defined: reward > limit (percentage); time too long
        with self.train_summary_writer.as_default():
            tf.summary.trace_export(
                name="veos_trace", step=epi_cnt_local, profiler_outdir=self.train_log_dir
            )
        thr_observe.join()
        thr_flash.join()

        # TODOt  test restore last table
        self.logc.info(f"Save the last table!!!!", extra=self.dictLogger)

        pds_last_table = pd.DataFrame(self.vcu_calib_table1, self.pd_index, self.pd_columns)

        last_table_store_path = (
            self.dataroot.joinpath(  #  there's no slash in the end of the string
                "last_table_ddpg-"
                + datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
                + ".csv"
            )
        )
        with open(last_table_store_path, "wb") as f:
            pds_last_table.to_csv(last_table_store_path)
        self.buffer.save()

        self.logc.info(f"main dies!!!!", extra=self.dictLogger)


if __name__ == "__main__":
    """
    ## Setup
    """
    # resumption settings
    parser = argparse.ArgumentParser(
        "use ddpg episodefree mode with tensorflow backend for EOS with coastdown activated and expected velocity in 3 seconds"
    )
    parser.add_argument(
        "-r",
        "--resume",
        default=False,
        help="resume the last training with restored model, checkpoint and pedal map",
        action="store_true",
    )

    parser.add_argument(
        "-i",
        "--infer",
        default=False,
        help="No model update and training. Only Inference",
        action="store_false",
    )
    parser.add_argument(
        "-t",
        "--record_table",
        default=True,
        help="record action table during training",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default=".",
        help="relative path to be saved, for create subfolder for different drivers",
    )
    args = parser.parse_args()

    # set up data folder (logging, checkpoint, table)

    app = realtime_train_infer_ddpg(
        args.resume, args.infer, args.record_table, args.path, projroot, logger
    )
    app.run()
