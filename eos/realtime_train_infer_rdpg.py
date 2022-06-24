"""
Title: realtime_train_infer_rdpg
Author: [Binjian Xin](https://www.newrizon.com)
Date created: 2022/05/16
Last modified: 2022/05/16
Description: Implement realtime RDPG algorithm for training and inference
"""
"""
## Introduction

This script shows an implementation of RDPG method on l045a truck real environment.

### Deep Deterministic Policy Gradient (RDPG) 

### Gym-Carla env 

An Ego Vehicle drives through a fixed track and collect loss (negative reward) defined
as energy consumption 

### References


"""

# system imports
import os
import argparse
import datetime

import socket
import json
import threading
import warnings

from threading import Lock, Thread
import time, queue, math, signal
from pathlib import PurePosixPath
# third party imports
from collections import deque
import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
from tensorflow.python.client import device_lib

# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# tf.debugging.set_log_device_placement(True)
## visualization import
import pandas as pd
import matplotlib.pyplot as plt

## logging
import logging
from logging.handlers import SocketHandler
import inspect
from pythonjsonlogger import jsonlogger

# local imports

from eos.visualization import plot_to_image, plot_3d_figure
from eos.comm import generate_vcu_calibration, kvaser_send_float_array, RemoteCan
from eos.agent import RDPG
from eos import logger, dictLogger, projroot

# from utils import get_logger, get_truck_status, flash_vcu, plot_3d_figure
# value = [99.0] * 21 * 17
# send_float_array('TQD_trqTrqSetECO_MAP_v', value)

# system warnings and numpy warnings handling
warnings.filterwarnings("ignore", message="currentThread", category=DeprecationWarning)
np.warnings.filterwarnings("ignore", category=DeprecationWarning)

# global variables: threading, data, lock, etc.
class realtime_train_infer_rdpg(object):
    def __init__(
        self,
        cloud=False,
        resume=False,
        infer=False,
        record=True,
        path=".",
        projroot=".",
        logger=None,
    ):
        self.cloud = cloud
        self.projroot = projroot
        self.logger = logger
        self.dictLogger = dictLogger
        # self.dictLogger = {"user": inspect.currentframe().f_code.co_name}
        self.resume = resume
        self.infer = infer
        self.record = record
        self.path = path

        self.eps = np.finfo(
            np.float32
        ).eps.item()  # smallest number such that 1.0 + eps != 1.0
        self.h_t = []

        if self.cloud:
            # reset proxy (internal site force no proxy)
            os.environ["http_proxy"] = ""
            self.client = RemoteCan(vin="987654321654321M4")
            self.get_truck_status = self.cloud_get_truck_status
        else:
            self.get_truck_status = self.onboard_get_truck_status

        if resume:
            self.dataroot = projroot.joinpath("data/" + self.path)
        else:
            self.dataroot = projroot.joinpath("data/scratch/" + self.path)

        self.set_logger()
        self.logger.info(f"Start Logging", extra=self.dictLogger)

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
        self.touch_gpu()
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
            "eos-rt-rdpg-"
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
        strfilename = PurePosixPath(logfilename).stem + ".json"
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
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = self.dataroot.joinpath(
            "tf_logs/rdpg/gradient_tape/" + current_time + "/train"
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
                self.projroot.joinpath("eos/config"),
            )
        self.vcu_calib_table1 = np.copy(
            self.vcu_calib_table0
        )  # shallow copy of the default table
        vcu_table1 = self.vcu_calib_table1.reshape(-1).tolist()
        self.logger.info(f"Start flash initial table", extra=self.dictLogger)
        # time.sleep(1.0)
        if self.cloud:
            self.can_client.send_torque_cmd(vcu_table1)
        else:
            returncode = kvaser_send_float_array(
                "TQD_trqTrqSetNormal_MAP_v", vcu_table1, sw_diff=False
            )

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
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """

        # create actor-critic network
        self.num_observations = 3  # observed are velocity, throttle, brake percentage; !! acceleration not available in l045a
        if self.cloud:
            self.observation_len = 50  # 50 observation tuples as a valid observation for agent, for period of 40ms, this is equal to 2 second
            self.sample_rate = 0.04  # sample rate of the observation tuples
        else:
            self.observation_len = 30  # 30 observation pairs as a valid observation for agent, for period of 50ms, this is equal to 1.5 second
            self.sample_rate = 0.05  # sample rate of the observation tuples
        self.num_inputs = (
            self.num_observations * self.observation_len
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

        # Learning rate for actor-critic models
        self.critic_lr = 0.002
        self.actor_lr = 0.001
        # Discount factor for future rewards
        self.gamma = 0.99
        # Used to update target networks
        self.tauAC = (0.001, 0.001)
        self.lrAC = (0.001, 0.002)
        self.seq_len = 8  # TODO  7 maximum sequence length
        self.buffer_capacity = 300000
        self.batch_size = 4
        # number of hidden units in the actor and critic networks
        self.hidden_unitsAC = (256, 256)
        # number of layer in the actor-critic network
        self.n_layerAC = (2, 2)
        # padding value for the input, impossible value for observation, action or reward
        self.padding_value = -10000
        self.ckpt_interval = 5

        # Initialize networks
        self.rdpg = RDPG(
            self.num_observations,
            self.observation_len,
            self.seq_len,
            self.num_reduced_actions,
            buffer_capacity=self.buffer_capacity,
            batch_size=self.batch_size,
            hidden_unitsAC=self.hidden_unitsAC,
            n_layersAC=self.n_layerAC,
            padding_value=self.padding_value,
            gamma=self.gamma,
            tauAC=self.tauAC,
            lrAC=self.lrAC,
            datafolder=str(self.dataroot),
            ckpt_interval=self.ckpt_interval,
        )

    def touch_gpu(self):

        # tf.summary.trace_on(graph=True, profiler=True)
        # ignites manual loading of tensorflow library, to guarantee the real-time processing of first data in main thread
        init_motionpower = np.random.rand(self.observation_len, self.num_observations)
        init_states = tf.convert_to_tensor(
            init_motionpower
        )  # state must have 30 (speed, throttle, current, voltage) 5 tuple
        input_array = tf.reshape(init_states, -1)
        # init_states = tf.expand_dims(input_array, 0)  # motion states is 30*2 matrix

        action0 = self.rdpg.actor_predict(input_array, 0)
        self.logger.info(
            f"manual load tf library by calling convert_to_tensor",
            extra=self.dictLogger,
        )
        self.rdpg.reset_noise()

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
            while not self.motionpowerQueue.empty():
                self.motionpowerQueue.get()
            self.logc.info("%s", "Episode done!!!", extra=self.dictLogger)
            self.vel_hist_dQ.clear()
            # raise Exception("reset capture to stop")
        self.logc.info(f"Coutndown dies!!!", extra=self.dictLogger)

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
        self.epi_countdown_time = (
            3  # extend capture time after valid episode temrination
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

    def onboard_get_truck_status(self, evt_epi_done):
        """
        This function is used to get the truck status
        from the onboard udp socket server of CAN capture module Kvaser
        """
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
                                    f"Cycle velocity: Aver{vel_aver:.2f},Min{vel_min:.2f},Max{vel_max:.2f},StartIndex{self.vcu_calib_table_row_start}!",
                                    extra=self.dictLogger,
                                )
                                # self.logd.info(
                                #     f"Producer Queue has {motionpowerQueue.qsize()}!", extra=self.dictLogger,
                                # )
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
                table = self.tableQueue.get(
                    block=False, timeout=1
                )  # default block = True
                # print("2 tablequeue size: {}".format(tablequeue.qsize()))
            except queue.Empty:
                pass
            else:

                # tf.print('calib table:', table, output_stream=output_path)
                self.logc.info(f"flash starts", extra=self.dictLogger)
                if self.cloud:
                    success, reson = self.client.send_torque_map(table)
                    if not success:
                        self.logc.error(
                            f"send_torque_map failed: {reson}",
                            extra=self.dictLogger,
                        )
                else:
                    returncode = kvaser_send_float_array(
                        "TQD_trqTrqSetNormal_MAP_v", table, sw_diff=True
                    )
                    if returncode != 0:
                        self.logc.error(
                            f"kvaser_send_float_array failed: {returncode}",
                            extra=self.dictLogger,
                        )
                # time.sleep(1.0)
                self.logc.info(
                    f"flash done, count:{flash_count}", extra=self.dictLogger
                )
                flash_count += 1
                # watch(flash_count)

        # motionpowerQueue.join()
        self.logc.info(f"flash_vcu dies!!!", extra=self.dictLogger)

    def cloud_get_truck_status(self):
        # global program_exit
        # global motionpowerQueue, sequence_len
        # global episode_count, episode_done, episode_end
        # global vcu_calib_table_row_start

        # self.logger.info(f'Start Initialization!', extra=self.dictLogger)
        start_moment = time.time()
        th_exit = False
        last_moment = time.time()
        # self.logc.info(f"Initialization Done!", extra=self.dictLogger)
        # qobject_size = 0

        self.vel_hist_dQ = deque(maxlen=25)  # accumulate 1s of velocity values
        # vel_cycle_dQ = deque(maxlen=30)  # accumulate 1.5s (one cycle) of velocity values
        vel_cycle_dQ = deque(
            maxlen=self.observation_len
        )  # accumulate 2s (one cycle) of velocity values

        duration = self.observation_len / self.sample_rate
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
            status_ok, remotecan_data = self.client.get_signals(duration=duration)
            if not status_ok:
                self.logc.error(
                    f"get_signals failed: {remotecan_data}",
                    extra=self.dictLogger,
                )
                continue
            # self.logger.info('Data received!!!', extra=self.dictLogger)
            data_type = type(remotecan_data)
            self.logc.info(f"Data type is {data_type}", extra=self.dictLogger)
            if not isinstance(remotecan_data, dict):
                self.logd.critical(
                    f"udp sending wrong data type!", extra=self.dictLogger
                )
                raise TypeError("udp sending wrong data type!")

            try:
                for key, value in remotecan_data.items():
                    if key == "status":  # state machine chores
                        # print(candata)
                        if value == "begin":
                            self.get_truck_status_start = True
                            self.logc.info(
                                "%s", "Episode will start!!!", extra=self.dictLogger
                            )
                            th_exit = False
                            # ts_epi_start = time.time()

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
                            self.get_truck_status_motpow_t = []
                            while not self.motionpowerQueue.empty():
                                self.motionpowerQueue.get()
                            self.logc.info(
                                "%s", "Episode done!!!", extra=self.dictLogger
                            )
                            th_exit = False
                            self.vel_hist_dQ.clear()
                            with self.hmi_lock:
                                self.episode_count += 1  # valid round increments
                                self.episode_done = True
                                self.episode_end = True
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
                                self.epi_countdown = False
                            break
                            # time.sleep(0.1)

                    elif key == "result":
                        # self.logger.info('Data received before Capture starting!!!', extra=self.dictLogger)
                        # self.logger.info(f'ts:{value["timestamp"]}vel:{value["velocity"]}ped:{value["pedal"]}', extra=self.dictLogger)
                        # DONE add logic for episode valid and invalid
                        try:
                            if self.get_truck_status_start:  # starts episode

                                # with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.1f}'.format}, linewidth=100):
                                with np.printoptions(suppress=True, linewidth=100):
                                    # capture warning about ragged json arrays
                                    with np.testing.suppress_warnings() as sup:
                                        log_warning = sup.record(
                                            np.VisibleDeprecationWarning,
                                            "Creating an ndarray from ragged nested sequences",
                                        )
                                        current = np.array(value["list_current_1s"])
                                        if len(log_warning) > 0:
                                            log_warning.pop()
                                            item_len = [len(item) for item in current]
                                            for count, item in enumerate(current):
                                                item[
                                                    item_len[count] : max(item_len)
                                                ] = None
                                        self.logd.info(
                                            f"current{current.shape}:{current}",
                                            extra=self.dictLogger,
                                        )

                                        voltage = np.array(value["list_voltage_1s"])
                                        if len(log_warning):
                                            log_warning.pop()
                                            item_len = [len(item) for item in voltage]
                                            for count, item in enumerate(voltage):
                                                item[
                                                    item_len[count] : max(item_len)
                                                ] = None
                                        # voltage needs to be upsampled in columns since its sample rate is half of others
                                        r_v, c_v = voltage.shape
                                        voltage_upsampled = np.empty(
                                            (r_v, 1, c_v, 2), dtype=voltage.dtype
                                        )
                                        voltage_upsampled[...] = voltage[
                                            :, None, :, None
                                        ]
                                        voltage = voltage_upsampled.reshape(
                                            r_v, c_v * 2
                                        )
                                        self.logd.info(
                                            f"voltage{voltage.shape}:{voltage}",
                                            extra=self.dictLogger,
                                        )

                                        thrust = np.array(value["list_pedal_1s"])
                                        if len(log_warning) > 0:
                                            log_warning.pop()
                                            item_len = [len(item) for item in thrust]
                                            for count, item in enumerate(thrust):
                                                item[
                                                    item_len[count] : max(item_len)
                                                ] = None
                                        self.logd.info(
                                            f"accl{thrust.shape}:{thrust}",
                                            extra=self.dictLogger,
                                        )

                                        brake = np.array(
                                            value["list_brake_pressure_1s"]
                                        )
                                        if len(log_warning) > 0:
                                            log_warning.pop()
                                            item_len = [len(item) for item in brake]
                                            for count, item in enumerate(brake):
                                                item[
                                                    item_len[count] : max(item_len)
                                                ] = None
                                        self.logd.info(
                                            f"brake{brake.shape}:{brake}",
                                            extra=self.dictLogger,
                                        )

                                        velocity = np.array(value["list_speed_1s"])
                                        if len(log_warning) > 0:
                                            log_warning.pop()
                                            item_len = [len(item) for item in velocity]
                                            for count, item in enumerate(velocity):
                                                item[
                                                    item_len[count] : max(item_len)
                                                ] = None
                                        self.logd.info(
                                            f"velocity{velocity.shape}:{velocity}",
                                            extra=self.dictLogger,
                                        )

                                        gears = np.array(value["list_gears"])
                                        if len(log_warning) > 0:
                                            log_warning.pop()
                                            item_len = [len(item) for item in gears]
                                            for count, item in enumerate(gears):
                                                item[
                                                    item_len[count] : max(item_len)
                                                ] = None
                                        # upsample gears from 2Hz to 25Hz
                                        r_v, c_v = gears.shape
                                        gears_upsampled = np.empty(
                                            (r_v, 1, c_v, 12), dtype=gears.dtype
                                        )
                                        gears_upsampled[...] = gears[:, None, :, None]
                                        gears = gears_upsampled.reshape(r_v, c_v * 12)
                                        gears = np.c_[
                                            gears, gears[:, -1]
                                        ]  # duplicate last gear on the end
                                        gears = gears.reshape(-1, 1)
                                        self.logd.info(
                                            f"gears{gears.shape}:{gears}",
                                            extra=self.dictLogger,
                                        )

                                        timestamp = np.array(value["timestamp"])
                                        self.logd.info(
                                            f"timestamp{timestamp.shape}:{datetime.fromtimestamp(timestamp.tolist())}",
                                            extra=self.dictLogger,
                                        )

                                motion_power = np.c_[
                                    velocity.reshape(-1, 1),
                                    thrust.reshape(-1, 1),
                                    brake.reshape(-1, 1),
                                    current.reshape(-1, 1),
                                    voltage.reshape(-1, 1),
                                ]  # 3 +2 : im 5

                                self.get_truck_status_motpow_t.append(
                                    motion_power
                                )  # obs_reward [speed, pedal, brake, current, voltage]
                                self.vel_hist_dQ.append(velocity)

                                vel_aver = velocity.mean()
                                vel_min = velocity.min()
                                vel_max = velocity.max()

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
                                self.motionpowerQueue.put(
                                    self.get_truck_status_motpow_t
                                )
                                self.get_truck_status_motpow_t = []
                        except Exception as X:
                            self.logc.info(
                                X,  # f"Valid episode, Reset data capturing to stop after 3 seconds!",
                                extra=self.dictLogger,
                            )
                    elif key == "reson":
                        self.logd.info(
                            f"reson: {value}",
                            extra=self.dictLogger,
                        )
                    elif key == "success":
                        self.logd.info(
                            f"success: {value}",
                            extra=self.dictLogger,
                        )
                    elif key == "elapsed":
                        self.logd.info(
                            f"elapsed: {value}",
                            extra=self.dictLogger,
                        )
                    else:
                        self.logc.critical(
                            "udp sending unknown signal (neither status nor data)!"
                        )
                        break
            except Exception as X:
                self.logc.info(f"{X}: data corrupt!", extra=self.dictLogger)

        self.logger.info(f"get_truck_status dies!!!", extra=self.dictLogger)

    # @eye
    def run(self):
        # global episode_count
        # global program_exit
        # global motionpowerQueue
        # global pd_index, pd_columns
        # global episode_done, episode_end
        # global vcu_calib_table_row_start

        # Start thread for flashing vcu, flash first
        evt_epi_done = threading.Event()
        thr_countdown = Thread(
            target=self.capture_countdown_handler, name="countdown", args=[evt_epi_done]
        )
        thr_observe = Thread(
            target=self.get_truck_status, name="observe", args=[evt_epi_done]
        )
        thr_flash = Thread(target=self.flash_vcu, name="flash", args=())
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
            wh1 = 0  # initialize odd step wh
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
                    # TODO l045a define round done (time, distance, defined end event)
                    with self.hmi_lock:  # wait for tester to interrupt or to exit
                        th_exit = (
                            self.program_exit
                        )  # if program_exit is False, reset to wait
                        epi_end = self.episode_end
                        done = self.episode_done  # this class member episode_done is driving action (maneuver) done
                        table_start = self.vcu_calib_table_row_start
                    self.logc.info(f"motionpowerQueue.qsize(): {self.motionpowerQueue.qsize()}")
                    if epi_end and done and (self.motionpowerQueue.qsize()>2):
                        # self.logc.info(f"motionpowerQueue.qsize(): {self.motionpowerQueue.qsize()}")
                        self.logc.info(f"Residue in Queue is a sign of disordered sequence, interrupted!")
                        done = False # this local done is true done with data exploitation
                        epi_end = True


                    if epi_end:  # stop observing and inferring
                        continue

                    try:
                        self.logc.info(
                            f"E{epi_cnt} Wait for an object!!!", extra=self.dictLogger
                        )
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
                    # with tf.device('/GPU:0'):
                    motpow_t = tf.convert_to_tensor(
                        motionpower
                    )  # state must have 30 (velocity, pedal, brake, current, voltage) 5 tuple (num_observations)
                    o_t0, pow_t = tf.split(motpow_t, [3, 2], 1)
                    o_t = tf.reshape(o_t0, -1)

                    self.logd.info(
                        f"E{epi_cnt} tensor convert and split!",
                        extra=self.dictLogger,
                    )
                    ui_sum = tf.reduce_sum(
                        tf.reduce_prod(pow_t, 1)
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
                        prev_r_t = (wh1 + wh) * (
                            -1.0
                        )  # most recent odd and even indexed reward
                        episode_reward += prev_r_t
                        # TODO add speed sum as positive reward

                        if step_count > 0:
                            if step_count == 2:  # first even step has $r_0$
                                self.h_t = [np.hstack([prev_o_t, prev_a_t, prev_r_t])]
                            else:
                                self.h_t.append(np.hstack([prev_o_t, prev_a_t, prev_r_t]))

                            self.logd.info(
                                f"prev_o_t.shape: {prev_o_t.shape},prev_a_t.shape: {prev_a_t.shape}, prev_r_t.shape: {prev_r_t.shape}, self.h_t shape: {len(self.h_t)}X{self.h_t[-1].shape}.",
                                extra=self.dictLogger,
                            )
                        # predict action probabilities and estimated future rewards
                        # from environment state
                        # for causal rl, the odd indexed observation/reward are caused by last action
                        # skip the odd indexed observation/reward for policy to make it causal
                        self.logc.info(
                            f"E{epi_cnt} before inference!",
                            extra=self.dictLogger,
                        )
                        # motion states o_t is 30*3/50*3 matrix
                        # with tf.device('/GPU:0'):
                        #     a_t = self.rdpg.actor_predict(o_t, step_count / 2)

                        a_t = self.rdpg.actor_predict(o_t, int(step_count / 2))
                        # self.logd.info(
                        #     f"E{epi_cnt} step{int(step_count/2)},o_t.shape:{o_t.shape},a_t.shape:{a_t.shape}!",
                        #     extra=self.dictLogger,
                        # )

                        prev_o_t = o_t
                        prev_a_t = a_t

                        self.logd.info(
                            f"E{epi_cnt} inference done with reduced action space!",
                            extra=self.dictLogger,
                        )

                        vcu_calib_table_reduced = tf.reshape(
                            a_t,
                            [
                                self.vcu_calib_table_row_reduced,
                                self.vcu_calib_table_col,
                            ],
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
                            table_start : self.vcu_calib_table_row_reduced
                            + table_start,
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
                            table_start : self.vcu_calib_table_row_reduced
                            + table_start,
                            :,
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
                                "tables/instant_table_rdpg-"
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

            # add episode history to agent replay buffer
            self.rdpg.add_to_replay(self.h_t)

            if self.infer:
                (actor_loss, critic_loss) = self.rdpg.notrain()
                self.logd.info("No Learning, just calculating loss")
            else:
                self.logd.info("Learning and soft updating")
                for k in range(6):
                    # self.logger.info(f"BP{k} starts.", extra=self.dictLogger)
                    (actor_loss, critic_loss) = self.rdpg.train()
                    # self.logd.info("Learning and soft updating")
                    # logd.info(f"BP{k} done.", extra=dictLogger)
                    self.logd.info(
                        f"E{epi_cnt}BP{k} critic loss: {critic_loss}; actor loss: {actor_loss}",
                        extra=dictLogger,
                    )
                    self.rdpg.soft_update_target()
                    # logger.info(f"Updated target critic.", extra=dictLogger)

                # Checkpoint manager save model
                self.rdpg.save_ckpt()

            self.logd.info(
                f"E{epi_cnt}BP{k} critic loss: {critic_loss}; actor loss: {actor_loss}",
                extra=self.dictLogger,
            )

            # update running reward to check condition for solving
            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

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
                    "Calibration Table Hist", vcu_act_list, step=epi_cnt_local
                )
                # tf.summary.trace_export(
                #     name="veos_trace", step=epi_cnt_local, profiler_outdir=self.train_log_dir
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
                "last_table_rdpg-"
                + datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
                + ".csv"
            )
        )
        with open(last_table_store_path, "wb") as f:
            pds_last_table.to_csv(last_table_store_path)

        self.rdpg.save_replay_buffer()
        self.logc.info(f"main dies!!!!", extra=self.dictLogger)


if __name__ == "__main__":
    """
    ## Setup
    """
    # resumption settings
    parser = argparse.ArgumentParser(
        "Use Recurrent DPG with tensorflow backend for EOS with coastdown activated and expected velocity in 3 seconds"
    )
    parser.add_argument(
        "-c",
        "--cloud",
        default=False,
        help="Use cloud mode, default is False",
        action="store_false",
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

    app = realtime_train_infer_rdpg(
        args.cloud,
        args.resume,
        args.infer,
        args.record_table,
        args.path,
        projroot,
        logger,
    )
    app.run()
