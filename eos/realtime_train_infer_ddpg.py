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

import argparse
import json

# logging
import logging
import math

# system imports
import os
import sys
import queue
import socket
import threading
import time
import warnings

# third party imports
from collections import deque
from datetime import datetime
from logging.handlers import SocketHandler
from pathlib import Path, PurePosixPath
from threading import Lock, Thread

# from bson import ObjectId

import matplotlib.pyplot as plt
import numpy as np

# tf.debugging.set_log_device_placement(True)
# visualization import
import pandas as pd
import tensorflow as tf
from pythonjsonlogger import jsonlogger

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
from tensorflow.python.client import device_lib

from eos import dictLogger, logger, projroot
from eos.agent import (
    Buffer,
    OUActionNoise,
    get_actor,
    get_critic,
    policy,
    update_target,
)
from eos.comm import RemoteCan, generate_vcu_calibration, kvaser_send_float_array
from eos.config import PEDAL_SCALES, trucks
from eos.utils import ragged_nparray_list_interp
from eos.visualization import plot_3d_figure, plot_to_image

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


# local imports


# from utils import get_logger, get_truck_status, flash_vcu, plot_3d_figure
# value = [99.0] * 21 * 17
# send_float_array('TQD_trqTrqSetECO_MAP_v', value)

# system warnings and numpy warnings handling
warnings.filterwarnings("ignore", message="currentThread", category=DeprecationWarning)
np.warnings.filterwarnings("ignore", category=DeprecationWarning)

# global variables: threading, data, lock, etc.


class RealtimeDDPG(object):
    def __init__(
        self,
        cloud=False,
        resume=False,
        infer=False,
        record=True,
        path=".",
        proj_root=Path("."),
        vlogger=None,
    ):
        self.cloud = cloud
        self.trucks = trucks
        self.truck_name = "VB7"  # 0: VB7, 1: VB6
        self.projroot = proj_root
        self.logger = vlogger
        self.dictLogger = dictLogger
        # self.dictLogger = {"user": inspect.currentframe().f_code.co_name}
        self.resume = resume
        self.infer = infer
        self.record = record
        self.path = path

        if resume:
            self.dataroot = projroot.joinpath("data/" + self.path)
        else:
            self.dataroot = projroot.joinpath("data/scratch/" + self.path)

        self.set_logger()
        self.logger.info(f"Start Logging", extra=self.dictLogger)

        # validate truck id
        # assert self.truck_name in self.trucks.keys()
        try:
            self.truck = self.trucks[self.truck_name]
        except KeyError as e:
            self.logger.error(
                f"{e}. No Truck with name {self.truck_name}", extra=self.dictLogger
            )
            sys.exit(1)

        # if self.truck.TruckName != "VB7":
        #     raise TruckIDError("Truck is not VB7")
        # else:
        #     self.logger.info(f"Truck is VB7", extra=self.dictLogger)

        self.eps = np.finfo(
            np.float32
        ).eps.item()  # smallest number such that 1.0 + eps != 1.0

        if self.cloud:
            # reset proxy (internal site force no proxy)
            self.init_cloud()
            self.get_truck_status = self.remote_hmi_state_machine
            self.flash_vcu = self.remote_flash_vcu
        else:
            self.get_truck_status = self.kvaser_get_truck_status
            self.flash_vcu = self.kvaser_flash_vcu

        self.logc.info(
            f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}"
        )
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
        self.touch_gpu()
        self.logger.info(f"VCU and GPU Initialization done!", extra=self.dictLogger)
        self.init_threads_data()
        self.logger.info(f"Thread data Initialization done!", extra=self.dictLogger)

    def init_cloud(self):
        os.environ["http_proxy"] = ""
        self.remotecan_client = RemoteCan(
            truckname=self.truck.TruckName, url="http://10.0.64.78:5000/"
        )

    def set_logger(self):
        self.logroot = self.dataroot.joinpath("py_logs")
        try:
            os.makedirs(self.logroot)
        except FileExistsError:
            print("User folder exists, just resume!")

        logfilename = self.logroot.joinpath(
            "eos-rt-ddpg-vb-" + datetime.now().isoformat().replace(":", "-") + ".log"
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
        # strfilename = PurePosixPath(logfilename).stem + ".json"
        strfilename = self.logroot.joinpath(PurePosixPath(logfilename).stem + ".json")
        strh = logging.FileHandler(strfilename, mode="a")
        strh.setLevel(logging.DEBUG)
        strh.setFormatter(json_file_formatter)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        #  Cutelog socket
        skh = SocketHandler("127.0.0.1", 19996)
        skh.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(strh)
        self.logger.addHandler(ch)
        self.logger.addHandler(skh)

        self.logger.setLevel(logging.DEBUG)
        # self.dictLogger = {'funcName': '__self__.__func__.__name__'}
        # self.dictLogger = {'user': inspect.currentframe().f_back.f_code.co_name}

        self.logc = logger.getChild("control flow")
        self.logc.propagate = True
        self.logd = logger.getChild("data flow")
        self.logd.propagate = True
        self.tflog = tf.get_logger()
        self.tflog.addHandler(fh)
        self.tflog.addHandler(ch)
        self.tflog.addHandler(skh)
        self.tflog.addHandler(strh)

    def set_data_path(self):
        # Create folder for ckpts loggings.
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = self.dataroot.joinpath(
            "tf_logs-vb/ddpg/gradient_tape/" + current_time + "/train"
        )
        self.train_summary_writer = tf.summary.create_file_writer(
            str(self.train_log_dir)
        )
        # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        if self.resume:
            self.logger.info(f"Resume last training", extra=self.dictLogger)
        else:
            self.logger.info(f"Start from scratch", extra=self.dictLogger)

    def init_vehicle(self):
        # resume last pedal map / scratch from default table

        # initialize pedal map parameters
        self.vcu_calib_table_col = (
            self.truck.PedalScale
        )  #  17 number of pedal steps, x direction
        self.vcu_calib_table_row = (
            self.truck.VelocityScale
        )  #  14 numnber of velocity steps, y direction
        self.vcu_calib_table_size = self.vcu_calib_table_row * self.vcu_calib_table_col
        self.action_budget = self.truck.ActionBudget  # action_budget 250 Nm
        self.action_lower = self.truck.ActionLowerBound  # 0.8
        self.action_upper = self.truck.ActionUpperBound  # 1.0
        self.action_bias = self.truck.ActionBias  # 0.0

        # index of the current pedal map is speed in kmph
        pd_ind = np.linspace(0, 120, self.vcu_calib_table_row - 1)
        self.pd_index = np.insert(pd_ind, 1, 7)  # insert 7 kmph
        self.pd_columns = np.array(PEDAL_SCALES)

        self.pedal_range = self.truck.PedalRange  # [0, 1.0]
        self.velocity_range = self.truck.VelocityRange  # [0, 120.0]

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
                self.projroot.joinpath("eos/config"),
            )
        # pandas deep copy of the default table (while numpy shallow copy is sufficient)
        self.vcu_calib_table1 = self.vcu_calib_table0.copy(deep=True)
        self.logger.info(f"Start flash initial table", extra=self.dictLogger)
        # time.sleep(1.0)
        if self.cloud:
            returncode = self.remotecan_client.send_torque_map(
                pedalmap=self.vcu_calib_table1, swap=False
            )  # 14 rows for whole map
        else:
            returncode = kvaser_send_float_array(self.vcu_calib_table1, sw_diff=False)

        self.logger.info(
            f"Done flash initial table. returncode: {returncode}", extra=self.dictLogger
        )

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
            self.num_observation (int): Dimension of each state
            self.num_actions (int): Dimension of each action
        """

        # create actor-critic network
        self.num_observations = self.truck.ObservationNumber
        # self.num_observations = 3  # observed are velocity, throttle, brake percentage; !! acceleration not available in l045a
        if self.cloud:
            self.observation_len = (
                self.truck.CloudUnitNumber * self.truck.CloudSignalFrequency
            )  # 250 observation tuples as a valid observation for agent, for period of 20ms, this is equal to 5 second
            # self.observation_len = 250  # 50 observation tuples as a valid observation for agent, for period of 40ms, this is equal to 2 second
            self.sample_rate = (
                1.0 / self.truck.CloudSignalFrequency
            )  # sample rate of the observation tuples
            # self.sample_rate = 0.02  # sample rate 20ms of the observation tuples
        else:
            self.observation_len = (
                self.truck.KvaserObservationNumber
            )  # 30 observation pairs as a valid observation for agent, for period of 50ms, this is equal to 1.5 second
            # self.observation_len = 30  # 30 observation pairs as a valid observation for agent, for period of 50ms, this is equal to 1.5 second
            self.sample_rate = (
                1.0 / self.truck.KvaserObservationFrequency
            )  # sample rate of the observation tuples
            # self.sample_rate = 0.05  # sample rate 50ms of the observation tuples
        self.num_inputs = (
            self.num_observations * self.observation_len
        )  # 60 subsequent observations
        self.num_actions = self.vcu_calib_table_size  # 17*14 = 238
        self.vcu_calib_table_row_reduced = 4  ## 0:5 adaptive rows correspond to low speed from  0~20, 7~25, 10~30, 15~35, etc  kmh  # overall action space is the whole table
        self.num_reduced_actions = (  # 0:4 adaptive rows correspond to low speed from  0~20, 7~30, 10~40, 20~50, etc  kmh  # overall action space is the whole table
            self.vcu_calib_table_row_reduced * self.vcu_calib_table_col
        )  # 4x17= 68
        # hyperparameters for DRL
        self.num_hidden = 256
        self.num_hidden0 = 16
        self.num_hidden1 = 32

        # DYNAMIC: need to adapt the pointer to change different roi of the pm, change the starting row index
        self.vcu_calib_table_row_start = 0

        # Initialize networks
        self.actor_model = get_actor(
            self.num_observations,
            self.num_reduced_actions,
            self.observation_len,
            self.num_hidden,
            self.action_bias,
        )

        self.critic_model = get_critic(
            self.num_observations,
            self.num_reduced_actions,
            self.observation_len,
            self.num_hidden0,
            self.num_hidden1,
            self.num_hidden,
        )

        # Initialize networks
        self.target_actor = get_actor(
            self.num_observations,
            self.num_reduced_actions,
            self.observation_len,
            self.num_hidden,
            self.action_bias,
        )

        self.target_critic = get_critic(
            self.num_observations,
            self.num_reduced_actions,
            self.observation_len,
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
            self.observation_len,
            self.num_reduced_actions,
            buffer_capacity=self.buffer_capacity,
            batch_size=self.batch_size,
            gamma=self.gamma,
            datafolder=str(self.dataroot),
            cloud=self.cloud,
        )

        # ou_noise is a row vector sdfof num_actions dimension
        self.ou_noise_std_dev = 0.2
        self.ou_noise = OUActionNoise(
            mean=np.zeros(self.num_reduced_actions),
            std_deviation=float(self.ou_noise_std_dev)
            * np.ones(self.num_reduced_actions),
        )

    def init_checkpoint(self):
        # add checkpoints manager
        if self.resume:
            checkpoint_actor_dir = self.dataroot.joinpath(
                "tf_ckpts-vb/l045a_ddpg_actor"
            )
            checkpoint_critic_dir = self.dataroot.joinpath(
                "tf_ckpts-vb/l045a_ddpg_critic"
            )
        else:
            checkpoint_actor_dir = self.dataroot.joinpath(
                "tf_ckpts-vb/l045a_ddpg_actor"
                + datetime.now().strftime("%y-%m-%d-%H-%M-%S")
            )
            checkpoint_critic_dir = self.dataroot.joinpath(
                "tf_ckpts-vb/l045a_ddpg_critic"
                + datetime.now().strftime("%y-%m-%d-%H-%M-%S")
            )
        try:
            os.makedirs(checkpoint_actor_dir)
            self.logger.info(
                "Actor folder doesn't exist. Created!", extra=self.dictLogger
            )
        except FileExistsError:
            self.logger.info("Actor folder exists, just resume!", extra=self.dictLogger)
        try:
            os.makedirs(checkpoint_critic_dir)
            self.logger.info(
                "Critic folder doesn't exist. Created!", extra=self.dictLogger
            )
        except FileExistsError:
            self.logger.info(
                "Critic folder exists, just resume!", extra=self.dictLogger
            )

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

    def touch_gpu(self):

        # tf.summary.trace_on(graph=True, profiler=True)
        # ignites manual loading of tensorflow library, to guarantee the real-time processing of first data in main thread
        init_motionpower = np.random.rand(self.observation_len, self.num_observations)
        init_states = tf.convert_to_tensor(
            init_motionpower
        )  # state must have 30 (speed, throttle, current, voltage) 5 tuple
        init_states = tf.expand_dims(init_states, 0)  # motion states is 30*2 matrix

        action0 = policy(self.actor_model, init_states, self.ou_noise)
        self.logger.info(
            f"manual load tf library by calling convert_to_tensor",
            extra=self.dictLogger,
        )
        self.ou_noise.reset()

        # warm up gpu training graph execution pipeline
        if self.buffer.buffer_counter != 0:
            if not self.infer:
                self.logger.info(
                    f"ddpg warm up training!",
                    extra=self.dictLogger,
                )

                (actor_loss, critic_loss) = self.buffer.learn()
                update_target(
                    self.target_actor.variables,
                    self.actor_model.variables,
                    self.tau,
                )
                # self.logger.info(f"Updated target actor", extra=self.dictLogger)
                update_target(
                    self.target_critic.variables,
                    self.critic_model.variables,
                    self.tau,
                )

                # self.logger.info(f"Updated target critic.", extra=self.dictLogger)
                self.logger.info(
                    f"ddpg warm up training done!",
                    extra=self.dictLogger,
                )

    # @eye
    # tracer.start()

    def capture_countdown_handler(self, evt_epi_done):

        th_exit = False
        while not th_exit:
            with self.hmi_lock:
                if self.program_exit:
                    th_exit = True
                    continue

            self.logger.info(f"wait for countdown", extra=self.dictLogger)
            evt_epi_done.wait()
            evt_epi_done.clear()
            # if episode is done, sleep for the extension time
            time.sleep(self.epi_countdown_time)
            # cancel wait as soon as waking up
            self.logger.info(f"finish countdown", extra=self.dictLogger)

            with self.hmi_lock:
                self.episode_count += 1  # valid round increments
                self.episode_done = (
                    True  # TODO delay episode_done to make main thread keep running
                )
                self.episode_end = True
                self.get_truck_status_start = False
            # move clean up under mutex to avoid competetion.
            self.get_truck_status_motpow_t = []
            with self.captureQ_lock:
                while not self.motionpowerQueue.empty():
                    self.motionpowerQueue.get()
            self.logc.info("%s", "Episode done!!!", extra=self.dictLogger)
            if self.cloud:
                self.vel_hist_dQ.clear()
            # raise Exception("reset capture to stop")
        self.logc.info(f"Coutndown dies!!!", extra=self.dictLogger)

    def init_threads_data(self):
        # multithreading initialization
        self.hmi_lock = Lock()
        self.tableQ_lock = Lock()
        self.captureQ_lock = Lock()
        self.remoteClient_lock = Lock()

        # tableQueue contains a table which is a list of type float
        self.tableQueue = queue.Queue()
        # motionpowerQueue contains a vcu states list with N(20) subsequent motion states + reward as observation
        self.motionpowerQueue = queue.Queue()

        # initial status of the switches
        self.program_exit = False
        self.episode_done = False
        self.episode_end = False
        self.episode_count = 0
        self.step_count = 0
        if self.cloud:
            self.epi_countdown_time = (
                self.truck.CloudUnitNumber
                * self.truck.CloudUnitDuration  # extend capture time after valid episode temrination
            )
        else:
            self.epi_countdown_time = (
                self.truck.KvaserCountdownTime  # extend capture time after valid episode temrination (3s)
            )

        # use timer object
        # self.timer_capture_countdown = threading.Timer(
        #     self.capture_countdown, self.capture_countdown_handler
        # )
        # signal.signal(signal.SIGALRM, self.reset_capture_handler)
        self.get_truck_status_start = False
        self.epi_countdown = False
        self.get_truck_status_motpow_t = []
        self.get_truck_status_myHost = "127.0.0.1"
        self.get_truck_status_myPort = 8002
        self.get_truck_status_qobject_len = 12  # sequence length 1.5*12s

    def kvaser_get_truck_status(self, evt_epi_done, evt_remote_get):
        """
        This function is used to get the truck status
        from the onboard udp socket server of CAN capture module Kvaser
        evt_remote_get is not used in this function for kvaser
        just to keep the uniform interface with remote_get_truck_status
        """

        th_exit = False

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        socket.socket.settimeout(s, None)
        s.bind((self.get_truck_status_myHost, self.get_truck_status_myPort))
        # s.listen(5)
        self.logc.info(f"Socket Initialization Done!", extra=self.dictLogger)

        self.vel_hist_dQ = deque(maxlen=20)  # accumulate 1s of velocity values
        # vel_cycle_dQ = deque(maxlen=30)  # accumulate 1.5s (one cycle) of velocity values
        vel_cycle_dQ = deque(
            maxlen=self.observation_len
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
            # self.logc.info(f"Data type is {data_type}", extra=self.dictLogger)
            if not isinstance(pop_data, dict):
                self.logd.critical(
                    f"udp sending wrong data type!", extra=self.dictLogger
                )
                raise TypeError("udp sending wrong data type!")

            for key, value in pop_data.items():
                if key == "status":  # state machine chores
                    # print(candata)
                    if value == "begin":
                        self.get_truck_status_start = True
                        self.logc.info(
                            "%s", "Episode will start!!!", extra=self.dictLogger
                        )
                        th_exit = False
                        # ts_epi_start = time.time()

                        with self.captureQ_lock:
                            while not self.motionpowerQueue.empty():
                                self.motionpowerQueue.get()
                        self.vel_hist_dQ.clear()
                        with self.hmi_lock:
                            self.episode_done = False
                            self.episode_end = False
                    elif value == "end_valid":
                        # DONE for valid end wait for another 2 queue objects (3 seconds) to get the last reward!
                        # cannot sleep the thread since data capturing in the same thread, use signal alarm instead
                        self.get_truck_status_start = (
                            True  # do not stopping data capture immediately
                        )

                        # set flag for countdown thread
                        evt_epi_done.set()
                        self.logc.info(f"Episode end starts countdown!")
                        with self.hmi_lock:
                            # self.episode_count += 1  # valid round increments self.epi_countdown = False
                            self.episode_done = False  # TODO delay episode_done to make main thread keep running
                            self.episode_end = False
                    elif value == "end_invalid":
                        self.get_truck_status_start = False
                        self.logc.info(
                            f"Episode is interrupted!!!", extra=self.dictLogger
                        )
                        self.get_truck_status_motpow_t = []
                        self.vel_hist_dQ.clear()
                        # motionpowerQueue.queue.clear()
                        # self.logc.info(
                        #     f"Episode motionpowerQueue has {motionpowerQueue.qsize()} states remaining",
                        #     extra=self.dictLogger,
                        # )
                        with self.captureQ_lock:
                            while not self.motionpowerQueue.empty():
                                self.motionpowerQueue.get()
                        # self.logc.info(
                        #     f"Episode motionpowerQueue gets cleared!", extra=self.dictLogger
                        # )
                        th_exit = False
                        with self.hmi_lock:
                            self.episode_done = False
                            self.episode_end = True
                            self.episode_count += 1  # invalid round increments
                    elif value == "exit":
                        self.get_truck_status_start = False
                        self.get_truck_status_motpow_t = []
                        self.vel_hist_dQ.clear()
                        with self.captureQ_lock:
                            while not self.motionpowerQueue.empty():
                                self.motionpowerQueue.get()
                        # self.logc.info("%s", "Program will exit!!!", extra=self.dictLogger)
                        th_exit = True
                        # for program exit, need to set episode states
                        # final change to inform main thread
                        with self.hmi_lock:
                            self.episode_done = False
                            self.episode_end = True
                            self.program_exit = True
                            self.episode_count += 1
                        evt_epi_done.set()
                        break
                        # time.sleep(0.1)
                elif key == "data":
                    # self.logger.info('Data received before Capture starting!!!', extra=self.dictLogger)
                    # self.logger.info(f'ts:{value["timestamp"]}vel:{value["velocity"]}ped:{value["pedal"]}', extra=self.dictLogger)
                    # DONE add logic for episode valid and invalid
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
                            self.vel_hist_dQ.append(velocity)
                            vel_cycle_dQ.append(velocity)

                            if (
                                len(self.get_truck_status_motpow_t)
                                >= self.observation_len
                            ):
                                if len(vel_cycle_dQ) != vel_cycle_dQ.maxlen:
                                    self.logc.warning(  # the recent 1.5s average velocity
                                        f"cycle deque is inconsistent!",
                                        extra=self.dictLogger,
                                    )

                                vel_aver = sum(vel_cycle_dQ) / vel_cycle_dQ.maxlen
                                vel_min = min(vel_cycle_dQ)
                                vel_max = max(vel_cycle_dQ)

                                # 0~20km/h; 7~30km/h; 10~40km/h; 20~50km/h; ...
                                # average concept
                                # 10; 18; 25; 35; 45; 55; 65; 75; 85; 95; 105
                                #   13; 18; 22; 27; 32; 37; 42; 47; 52; 57; 62;
                                # here upper bound rule adopted
                                if vel_max < 20:
                                    self.vcu_calib_table_row_start = 0
                                elif vel_max < 30:
                                    self.vcu_calib_table_row_start = 1
                                elif vel_max < 120:
                                    self.vcu_calib_table_row_start = (
                                        math.floor((vel_max - 30) / 10) + 2
                                    )
                                else:
                                    self.logc.warning(
                                        f"cycle higher than 120km/h!",
                                        extra=self.dictLogger,
                                    )
                                    self.vcu_calib_table_row_start = 16
                                # get the row of the table

                                self.logd.info(
                                    f"Cycle velocity: Aver{vel_aver:.2f},Min{vel_min:.2f},Max{vel_max:.2f},StartIndex{self.vcu_calib_table_row_start}!",
                                    extra=self.dictLogger,
                                )
                                # self.logd.info(
                                #     f"Producer Queue has {motionpowerQueue.qsize()}!", extra=self.dictLogger,
                                # )

                                with self.captureQ_lock:
                                    self.motionpowerQueue.put(
                                        self.get_truck_status_motpow_t
                                    )
                                self.get_truck_status_motpow_t = []
                    except Exception as X:
                        self.logc.info(
                            X,  # f"Valid episode, Reset data capturing to stop after 3 seconds!",
                            extra=self.dictLogger,
                        )
                        break
                else:
                    self.logc.warning(
                        f"udp sending message with key: {key}; value: {value}!!!"
                    )

                    break

        self.logger.info(f"get_truck_status dies!!!", extra=self.dictLogger)

        s.close()

    # this is the calibration table consumer for flashing
    # @eye
    def kvaser_flash_vcu(self):

        flash_count = 0
        th_exit = False

        self.logc.info(f"Initialization Done!", extra=self.dictLogger)
        while not th_exit:
            # time.sleep(0.1)
            with self.hmi_lock:
                table_start = self.vcu_calib_table_row_start
                epi_cnt = self.episode_count
                step_count = self.step_count
                if self.program_exit:
                    th_exit = True
                    continue
            try:
                # print("1 tablequeue size: {}".format(tablequeue.qsize()))
                with self.tableQ_lock:
                    table = self.tableQueue.get(
                        block=False, timeout=1
                    )  # default block = True
                    # print("2 tablequeue size: {}".format(tablequeue.qsize()))
            except queue.Empty:
                pass
            else:

                vcu_calib_table_reduced = tf.reshape(
                    table,
                    [
                        self.vcu_calib_table_row_reduced,
                        self.vcu_calib_table_col,
                    ],
                )

                # get change budget : % of initial table
                vcu_calib_table_reduced = vcu_calib_table_reduced * self.action_budget

                # dynamically change table row start index
                vcu_calib_table0_reduced = self.vcu_calib_table0.to_numpy()[
                    table_start : self.vcu_calib_table_row_reduced + table_start,
                    :,
                ]
                vcu_calib_table_min_reduced = (
                    vcu_calib_table0_reduced - self.action_budget
                )
                vcu_calib_table_max_reduced = 1.0 * vcu_calib_table0_reduced

                vcu_calib_table_reduced = tf.clip_by_value(
                    vcu_calib_table_reduced + vcu_calib_table0_reduced,
                    clip_value_min=vcu_calib_table_min_reduced,
                    clip_value_max=vcu_calib_table_max_reduced,
                )

                # create updated complete pedal map, only update the first few rows
                # vcu_calib_table1 keeps changing as the cache of the changing pedal map
                self.vcu_calib_table1.iloc[
                    table_start : self.vcu_calib_table_row_reduced + table_start
                ] = vcu_calib_table_reduced.numpy()

                if args.record_table:
                    curr_table_store_path = self.dataroot.joinpath(
                        "tables/instant_table_ddpg-vb-"
                        + datetime.now().strftime("%y-%m-%d-%h-%m-%s-")
                        + "e-"
                        + str(epi_cnt)
                        + "-"
                        + str(step_count)
                        + ".csv"
                    )
                    with open(curr_table_store_path, "wb") as f:
                        self.vcu_calib_table1.to_csv(curr_table_store_path)
                        # np.save(last_table_store_path, vcu_calib_table1)
                    self.logd.info(
                        f"E{epi_cnt} done with record instant table: {step_count}",
                        extra=self.dictLogger,
                    )

                self.logc.info(f"flash starts", extra=self.dictLogger)
                returncode = kvaser_send_float_array(
                    self.vcu_calib_table1, sw_diff=True
                )
                # time.sleep(1.0)

                if returncode != 0:
                    self.logc.error(
                        f"kvaser_send_float_array failed: {returncode}",
                        extra=self.dictLogger,
                    )
                else:
                    self.logc.info(
                        f"flash done, count:{flash_count}", extra=self.dictLogger
                    )
                    flash_count += 1
                # watch(flash_count)

        self.logc.info(f"flash_vcu dies!!!", extra=self.dictLogger)

    def remote_get_handler(self, evt_remote_get):

        th_exit = False
        while not th_exit:
            with self.hmi_lock:
                if self.program_exit:
                    th_exit = True
                    continue

            self.logger.info(f"wait for remote get trigger", extra=self.dictLogger)
            evt_remote_get.wait()

            # if episode is done, sleep for the extension time
            # cancel wait as soon as waking up
            self.logger.info(f"Wake up to fetch remote data", extra=self.dictLogger)

            try:
                with self.remoteClient_lock:
                    (
                        signal_success,
                        remotecan_data,
                    ) = self.remotecan_client.get_signals(
                        duration=self.truck.CloudUnitNumber
                    )

                if not isinstance(remotecan_data, dict):
                    self.logd.critical(
                        f"udp sending wrong data type!",
                        extra=self.dictLogger,
                    )
                    raise TypeError("udp sending wrong data type!")

                if signal_success == 0:
                    try:
                        signal_freq = self.truck.CloudSignalFrequency
                        gear_freq = self.truck.CloudGearFrequency
                        unit_duration = self.truck.CloudUnitDuration
                        unit_ob_num = unit_duration * signal_freq
                        unit_gear_num = unit_duration * gear_freq
                        unit_num = self.truck.CloudUnitNumber
                        for key, value in remotecan_data.items():
                            if key == "result":
                                self.logd.info(
                                    "convert observation state to array.",
                                    extra=self.dictLogger,
                                )
                                # timestamp processing
                                timestamps = []
                                separators = "--T::."  # adaption separators of the raw intest string
                                start_century = "20"
                                timezone = "+0800"
                                for ts in value["timestamps"]:
                                    # create standard iso string datetime format
                                    ts_substrings = [
                                        ts[i : i + 2] for i in range(0, len(ts), 2)
                                    ]
                                    ts_iso = start_century
                                    for i, sep in enumerate(separators):
                                        ts_iso = ts_iso + ts_substrings[i] + sep
                                    ts_iso = ts_iso + ts_substrings[-1] + timezone
                                    timestamps.append(ts_iso)
                                timestamps_units = (
                                    np.array(timestamps)
                                    .astype("datetime64[ms]")
                                    .astype("int")  # convert to int
                                )
                                if len(timestamps_units) != unit_num:
                                    raise ValueError(
                                        f"timestamps_units length is {len(timestamps_units)}, not {unit_num}"
                                    )
                                # upsample gears from 2Hz to 50Hz
                                timestamps_seconds = list(timestamps_units)  # in ms
                                sampling_interval = 1.0 / signal_freq * 1000  # in ms
                                timestamps = [
                                    i + j * sampling_interval
                                    for i in timestamps_seconds
                                    for j in np.arange(unit_ob_num)
                                ]
                                timestamps = np.array(timestamps).reshape(
                                    (self.truck.CloudUnitNumber, -1)
                                )
                                current = ragged_nparray_list_interp(
                                    value["list_current_1s"],
                                    ob_num=unit_ob_num,
                                )
                                voltage = ragged_nparray_list_interp(
                                    value["list_voltage_1s"],
                                    ob_num=unit_ob_num,
                                )
                                thrust = ragged_nparray_list_interp(
                                    value["list_pedal_1s"],
                                    ob_num=unit_ob_num,
                                )
                                brake = ragged_nparray_list_interp(
                                    value["list_brake_pressure_1s"],
                                    ob_num=unit_ob_num,
                                )
                                velocity = ragged_nparray_list_interp(
                                    value["list_speed_1s"],
                                    ob_num=unit_ob_num,
                                )
                                gears = ragged_nparray_list_interp(
                                    value["list_gears"],
                                    ob_num=unit_gear_num,
                                )
                                # upsample gears from 2Hz to 50Hz
                                gears = np.repeat(
                                    gears,
                                    (signal_freq // gear_freq),
                                    axis=1,
                                )

                                motion_power = np.c_[
                                    timestamps.reshape(-1, 1),
                                    velocity.reshape(-1, 1),
                                    thrust.reshape(-1, 1),
                                    brake.reshape(-1, 1),
                                    gears.reshape(-1, 1),
                                    current.reshape(-1, 1),
                                    voltage.reshape(-1, 1),
                                ]  # 1 + 3 + 1 + 2  : im 7

                                # 0~20km/h; 7~30km/h; 10~40km/h; 20~50km/h; ...
                                # average concept
                                # 10; 18; 25; 35; 45; 55; 65; 75; 85; 95; 105
                                #   13; 18; 22; 27; 32; 37; 42; 47; 52; 57; 62;
                                # here upper bound rule adopted
                                vel_max = np.amax(velocity)
                                if vel_max < 20:
                                    self.vcu_calib_table_row_start = 0
                                elif vel_max < 30:
                                    self.vcu_calib_table_row_start = 1
                                elif vel_max < 120:
                                    self.vcu_calib_table_row_start = (
                                        math.floor((vel_max - 30) / 10) + 2
                                    )
                                else:
                                    self.logc.warning(
                                        f"cycle higher than 120km/h!",
                                        extra=self.dictLogger,
                                    )
                                    self.vcu_calib_table_row_start = 16

                                self.logd.info(
                                    f"Cycle velocity: Aver{np.mean(velocity):.2f},Min{np.amin(velocity):.2f},Max{np.amax(velocity):.2f},StartIndex{self.vcu_calib_table_row_start}!",
                                    extra=self.dictLogger,
                                )

                                with self.captureQ_lock:
                                    self.motionpowerQueue.put(motion_power)
                            else:
                                self.logger.info(
                                    f"show status: {key}:{value}",
                                    extra=self.dictLogger,
                                )

                    except Exception as X:
                        self.logger.error(
                            f"show status: exception {X}, data corruption.",
                            extra=self.dictLogger,
                        )
                else:
                    self.logd.error(
                        f"get_signals failed: {remotecan_data}",
                        extra=self.dictLogger,
                    )

            except Exception as X:
                self.logc.info(
                    X,  # f"Valid episode, Reset data capturing to stop after 3 seconds!",
                    extra=self.dictLogger,
                )
                break

            self.logc.info(f"Get on record!!!", extra=self.dictLogger)
            evt_remote_get.clear()

        self.logc.info(f"thr_remoteget dies!!!!!", extra=self.dictLogger)

    def remote_hmi_state_machine(self, evt_epi_done, evt_remote_get):
        """
        This function is used to get the truck status
        from remote can module
        """

        th_exit = False

        #  Get the HMI control command from UDP, but not the data from KvaserCAN
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        socket.socket.settimeout(s, None)
        s.bind((self.get_truck_status_myHost, self.get_truck_status_myPort))
        # s.listen(5)
        self.logc.info(f"Socket Initialization Done!", extra=self.dictLogger)

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
            # self.logc.info(f"Data type is {data_type}", extra=self.dictLogger)
            if not isinstance(pop_data, dict):
                self.logd.critical(
                    f"udp sending wrong data type!", extra=self.dictLogger
                )
                raise TypeError("udp sending wrong data type!")

            for key, value in pop_data.items():
                if key == "status":  # state machine chores
                    # print(candata)
                    if value == "begin":
                        self.get_truck_status_start = True
                        self.logc.info(
                            "%s", "Episode will start!!!", extra=self.dictLogger
                        )
                        th_exit = False
                        # ts_epi_start = time.time()

                        with self.captureQ_lock:
                            while not self.motionpowerQueue.empty():
                                self.motionpowerQueue.get()
                        with self.hmi_lock:
                            self.episode_done = False
                            self.episode_end = False
                    elif value == "end_valid":
                        # DONE for valid end wait for another 2 queue objects (3 seconds) to get the last reward!
                        # cannot sleep the thread since data capturing in the same thread, use signal alarm instead
                        self.get_truck_status_start = (
                            True  # do not stopping data capture immediately
                        )

                        # set flag for countdown thread
                        evt_epi_done.set()
                        self.logc.info(f"Episode end starts countdown!")
                        with self.hmi_lock:
                            # self.episode_count += 1  # valid round increments self.epi_countdown = False
                            self.episode_done = False  # TODO delay episode_done to make main thread keep running
                            self.episode_end = False
                    elif value == "end_invalid":
                        self.get_truck_status_start = False
                        self.logc.info(
                            f"Episode is interrupted!!!", extra=self.dictLogger
                        )
                        self.get_truck_status_motpow_t = []
                        # motionpowerQueue.queue.clear()
                        # self.logc.info(
                        #     f"Episode motionpowerQueue has {motionpowerQueue.qsize()} states remaining",
                        #     extra=self.dictLogger,
                        # )
                        with self.captureQ_lock:
                            while not self.motionpowerQueue.empty():
                                self.motionpowerQueue.get()
                        # self.logc.info(
                        #     f"Episode motionpowerQueue gets cleared!", extra=self.dictLogger
                        # )
                        th_exit = False
                        with self.hmi_lock:
                            self.episode_done = False
                            self.episode_end = True
                            self.episode_count += 1  # invalid round increments
                    elif value == "exit":
                        self.get_truck_status_start = False
                        self.get_truck_status_motpow_t = []

                        with self.captureQ_lock:
                            while not self.motionpowerQueue.empty():
                                self.motionpowerQueue.get()
                        # self.logc.info("%s", "Program will exit!!!", extra=self.dictLogger)
                        th_exit = True
                        # for program exit, need to set episode states
                        # final change to inform main thread
                        with self.hmi_lock:
                            self.episode_done = False
                            self.episode_end = True
                            self.program_exit = True
                            self.episode_count += 1
                        evt_epi_done.set()
                        break
                        # time.sleep(0.1)
                elif key == "data":
                    #  instead of get kvasercan, we get remotecan data here!
                    if self.get_truck_status_start:  # starts episode
                        # set flag for remote_get thread
                        evt_remote_get.set()
                        self.logc.info(f"Kick off remoteget!!")
                else:
                    self.logc.warning(
                        f"udp sending message with key: {key}; value: {value}!!!"
                    )

                    break

        s.close()
        self.logger.info(f"get_truck_status dies!!!", extra=self.dictLogger)

    def remote_flash_vcu(self):

        flash_count = 0
        th_exit = False

        self.logc.info(f"Initialization Done!", extra=self.dictLogger)
        while not th_exit:
            # time.sleep(0.1)
            with self.hmi_lock:
                table_start = self.vcu_calib_table_row_start
                epi_cnt = self.episode_count
                step_count = self.step_count
                if self.program_exit:
                    th_exit = True
                    continue
            try:
                # print("1 tablequeue size: {}".format(tablequeue.qsize()))
                with self.tableQ_lock:
                    table = self.tableQueue.get(
                        block=False, timeout=1
                    )  # default block = True
                    # print("2 tablequeue size: {}".format(tablequeue.qsize()))
            except queue.Empty:
                pass
            else:
                vcu_calib_table_reduced = tf.reshape(
                    table,
                    [
                        self.vcu_calib_table_row_reduced,
                        self.vcu_calib_table_col,
                    ],
                )

                # get change budget : % of initial table
                vcu_calib_table_reduced = vcu_calib_table_reduced * self.action_budget

                # dynamically change table row start index
                vcu_calib_table0_reduced = self.vcu_calib_table0.to_numpy()[
                    table_start : self.vcu_calib_table_row_reduced + table_start,
                    :,
                ]
                vcu_calib_table_min_reduced = (
                    vcu_calib_table0_reduced - self.action_budget
                )
                vcu_calib_table_max_reduced = 1.0 * vcu_calib_table0_reduced

                vcu_calib_table_reduced = tf.clip_by_value(
                    vcu_calib_table_reduced + vcu_calib_table0_reduced,
                    clip_value_min=vcu_calib_table_min_reduced,
                    clip_value_max=vcu_calib_table_max_reduced,
                )

                # create updated complete pedal map, only update the first few rows
                # vcu_calib_table1 keeps changing as the cache of the changing pedal map
                self.vcu_calib_table1.iloc[
                    table_start : self.vcu_calib_table_row_reduced + table_start
                ] = vcu_calib_table_reduced.numpy()

                if args.record_table:
                    curr_table_store_path = self.dataroot.joinpath(
                        "tables/instant_table_ddpg-vb-"
                        + datetime.now().strftime("%y-%m-%d-%h-%m-%s-")
                        + "e-"
                        + str(epi_cnt)
                        + "-"
                        + str(step_count)
                        + ".csv"
                    )
                    with open(curr_table_store_path, "wb") as f:
                        self.vcu_calib_table1.to_csv(curr_table_store_path)
                        # np.save(last_table_store_path, vcu_calib_table1)
                    self.logd.info(
                        f"E{epi_cnt} done with record instant table: {step_count}",
                        extra=self.dictLogger,
                    )

                self.logc.info(f"flash starts", extra=self.dictLogger)

                with self.remoteClient_lock:
                    returncode = self.remotecan_client.send_torque_map(
                        pedalmap=self.vcu_calib_table1.iloc[
                            table_start : self.vcu_calib_table_row_reduced + table_start
                        ],
                        swap=False,
                    )
                # time.sleep(1.0)

                if returncode != 0:
                    self.logc.error(
                        f"send_torque_map failed: {returncode}",
                        extra=self.dictLogger,
                    )
                else:
                    self.logc.info(
                        f"flash done, count:{flash_count}", extra=self.dictLogger
                    )
                    flash_count += 1
                # watch(flash_count)

        # motionpowerQueue.join()
        self.logc.info(f"remote_flash_vcu dies!!!", extra=self.dictLogger)

    # @eye
    def run(self):

        # Start thread for flashing vcu, flash first
        evt_epi_done = threading.Event()
        evt_remote_get = threading.Event()
        thr_countdown = Thread(
            target=self.capture_countdown_handler, name="countdown", args=[evt_epi_done]
        )
        thr_observe = Thread(
            target=self.get_truck_status,
            name="observe",
            args=[evt_epi_done, evt_remote_get],
        )

        thr_remoteget = Thread(
            target=self.remote_get_handler, name="remoteget", args=[]
        )
        thr_flash = Thread(target=self.flash_vcu, name="flash", args=[evt_remote_get])
        thr_countdown.start()
        thr_observe.start()
        thr_flash.start()

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
            wh0 = 0  # initialize odd step wh
            # tf.summary.trace_on(graph=True, profiler=True)

            self.logc.info("----------------------", extra=self.dictLogger)
            self.logc.info(
                f"E{epi_cnt} starts!",
                extra=self.dictLogger,
            )

            tf.debugging.set_log_device_placement(True)
            with tf.device("/GPU:0"):
                while (
                    not epi_end
                ):  # end signal, either the round ends normally or user interrupt
                    with self.hmi_lock:  # wait for tester to interrupt or to exit
                        th_exit = (
                            self.program_exit
                        )  # if program_exit is False, reset to wait
                        epi_end = self.episode_end
                        done = (
                            self.episode_done
                        )  # this class member episode_done is driving action (maneuver) done
                        table_start = self.vcu_calib_table_row_start
                        self.step_count = step_count

                    with self.captureQ_lock:
                        motionpowerqueue_size = self.motionpowerQueue.qsize()
                    self.logc.info(f"motionpowerQueue.qsize(): {motionpowerqueue_size}")
                    if epi_end and done and (motionpowerqueue_size > 2):
                        # self.logc.info(f"motionpowerQueue.qsize(): {self.motionpowerQueue.qsize()}")
                        self.logc.info(
                            f"Residue in Queue is a sign of disordered sequence, interrupted!"
                        )
                        done = (
                            False  # this local done is true done with data exploitation
                        )
                        epi_end = True

                    if epi_end:  # stop observing and inferring
                        continue

                    try:
                        self.logc.info(
                            f"E{epi_cnt} Wait for an object!!!", extra=self.dictLogger
                        )

                        with self.captureQ_lock:
                            motionpower = self.motionpowerQueue.get(
                                block=True, timeout=1.55
                            )
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
                    )  # state must have 30/100 (velocity, pedal, brake, current, voltage) 5 tuple (num_observations)
                    if self.cloud:
                        out = tf.split(
                            motionpower_states, [1, 3, 1, 2], 1
                        )  # note the difference of split between np and tf
                        (timstamp, motion_states, gear_states, power_states) = [
                            tf.squeeze(x) for x in out
                        ]
                    else:
                        motion_states, power_states = tf.split(
                            motionpower_states, [3, 2], 1
                        )  # note the difference of split between np and tf

                    self.logd.info(
                        f"E{epi_cnt} tensor convert and split!",
                        extra=self.dictLogger,
                    )
                    ui_sum = tf.reduce_sum(
                        tf.reduce_prod(power_states, 1)
                    )  # vcu reward is a scalar
                    wh = (
                        ui_sum / 3600.0 * self.sample_rate
                    )  # rate 0.05 for kvaser, 0.02 remote # negative wh
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
                        wh0 = wh
                        # motion_states_history.append(motion_states)
                        if (
                            step_count != 0
                        ):  # not the first step, starting from the second step
                            prev_motion_states = motion_states_even
                            prev_timestamp = timestamp_even
                            prev_table_start = table_start
                            if self.cloud:
                                prev_action = vcu_action_reduced.numpy().tolist()
                            else:
                                prev_action = vcu_action_reduced

                        motion_states_even = motion_states
                        timestamp_even = timstamp

                        motion_states0 = tf.expand_dims(
                            motion_states_even, 0
                        )  # motion states is 30*3 matrix

                        # predict action probabilities and estimated future rewards
                        # from environment state
                        # for causal rl, the odd indexed observation/reward are caused by last action
                        # skip the odd indexed observation/reward for policy to make it causal
                        self.logc.info(
                            f"E{epi_cnt} before inference!",
                            extra=self.dictLogger,
                        )
                        vcu_action_reduced = policy(
                            self.actor_model, motion_states0, self.ou_noise
                        )

                        self.logd.info(
                            f"E{epi_cnt} inference done with reduced action space!",
                            extra=self.dictLogger,
                        )

                        # tf.print('calib table:', vcu_act_list, output_stream=sys.stderr)
                        with self.tableQ_lock:
                            self.tableQueue.put(vcu_action_reduced)
                            self.logd.info(
                                f"E{epi_cnt} StartIndex {table_start} Action Push table: {self.tableQueue.qsize()}",
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

                        cycle_reward = (wh0 + wh) * (
                            -1.0
                        )  # most recent odd and even indexed reward
                        if step_count != 1:  # starting from 3rd step
                            episode_reward += cycle_reward

                            if self.cloud:
                                self.rec = {
                                    "timestamp": datetime.fromtimestamp(
                                        prev_timestamp.numpy()[0] / 1000.0
                                    ),  # from ms to s
                                    "plot": {
                                        "character": self.truck.TruckName,
                                        "when": datetime.fromtimestamp(
                                            prev_timestamp.numpy()[0] / 1000.0
                                        ),
                                        "where": "campus",
                                        "states": {
                                            "velocity_unit": "kmph",
                                            "thrust_unit": "percentage",
                                            "brake_unit": "percentage",
                                            "length": motion_states_even.shape[0],
                                        },
                                        "actions": {
                                            "action_row_number": self.vcu_calib_table_row_reduced,
                                            "action_column_number": self.vcu_calib_table_col,
                                            "action_start_row": prev_table_start,
                                        },
                                        "reward": {
                                            "reward_unit": "wh",
                                        },
                                    },
                                    "observation": {
                                        "state": prev_motion_states.numpy().tolist(),
                                        "action": prev_action,
                                        "reward": cycle_reward.numpy().tolist(),
                                        "next_state": motion_states_even.numpy().tolist(),
                                    },
                                }
                                self.buffer.deposit(self.rec)
                            else:
                                self.buffer.record(
                                    (
                                        prev_motion_states,
                                        prev_action,
                                        cycle_reward,
                                        motion_states,
                                    )
                                )

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

            critic_loss = 0
            actor_loss = 0
            if self.infer:
                (critic_loss, actor_loss) = self.buffer.nolearn()
                self.logd.info("No Learning, just calculating loss")
            else:
                self.logd.info("Learning and updating 6 times!")
                for k in range(6):
                    # self.logger.info(f"BP{k} starts.", extra=self.dictLogger)
                    (critic_loss, actor_loss) = self.buffer.learn()

                    update_target(
                        self.target_actor.variables,
                        self.actor_model.variables,
                        self.tau,
                    )
                    # self.logger.info(f"Updated target actor", extra=self.dictLogger)
                    update_target(
                        self.target_critic.variables,
                        self.critic_model.variables,
                        self.tau,
                    )
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
                f"E{epi_cnt}BP 6 times critic loss: {critic_loss}; actor loss: {actor_loss}",
                extra=self.dictLogger,
            )

            # update running reward to check condition for solving
            running_reward = 0.05 * (-episode_reward) + (1 - 0.05) * running_reward

            # Create a matplotlib 3d figure, //export and save in log
            fig = plot_3d_figure(self.vcu_calib_table1, self.pd_columns, self.pd_index)

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
                    "Calibration Table Hist", self.vcu_calib_table1, step=epi_cnt_local
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
        # with self.train_summary_writer.as_default():
        #     tf.summary.trace_export(
        #         name="veos_trace",
        #         step=epi_cnt_local,
        #         profiler_outdir=self.train_log_dir,
        #     )
        thr_countdown.join()
        thr_observe.join()
        thr_flash.join()

        # TODOt  test restore last table
        self.logc.info(f"Save the last table!!!!", extra=self.dictLogger)

        pds_last_table = pd.DataFrame(
            self.vcu_calib_table1, self.pd_index, self.pd_columns
        )

        last_table_store_path = (
            self.dataroot.joinpath(  #  there's no slash in the end of the string
                "last_table_ddpg-"
                + datetime.now().strftime("%y-%m-%d-%H-%M-%S")
                + ".csv"
            )
        )
        with open(last_table_store_path, "wb") as f:
            pds_last_table.to_csv(last_table_store_path)

        if self.cloud is False:
            self.buffer.save()
        #  for database, just exit no need to cleanup.

        self.logc.info(f"main dies!!!!", extra=self.dictLogger)


if __name__ == "__main__":
    """
    ## Setup
    """
    # resumption settings
    parser = argparse.ArgumentParser(
        "Use DDPG mode with tensorflow backend for EOS with coastdown activated and expected velocity in 3 seconds"
    )
    parser.add_argument(
        "-c",
        "--cloud",
        default=False,
        help="Use cloud mode, default is False",
        action="store_true",
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

    # try:
    app = RealtimeDDPG(
        args.cloud,
        args.resume,
        args.infer,
        args.record_table,
        args.path,
        projroot,
        logger,
    )
    # except TypeError as e:
    #     logger.error(f"Project Exeception TypeError: {e}", extra=dictLogger)
    #     sys.exit(1)
    # except Exception as e:
    #     logger.error(e, extra=dictLogger)
    #     sys.exit(1)
    app.run()
