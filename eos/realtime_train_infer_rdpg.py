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

import argparse
import json

# logging
import logging
import math

# system imports
import os
import queue
import socket
import sys
import subprocess
import threading
import time
import warnings
import re

# third party imports
from collections import deque
from datetime import datetime
from logging.handlers import SocketHandler
from pathlib import Path, PurePosixPath
from threading import Lock, Thread

import matplotlib.pyplot as plt
import numpy as np

# tf.debugging.set_log_device_placement(True)
# visualization import
import tensorflow as tf
from git import Repo
from pythonjsonlogger import jsonlogger

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
from tensorflow.python.client import device_lib
from rocketmq.client import Message, Producer

from eos import dictLogger, logger, projroot
from eos.agent import RDPG
from eos.comm import RemoteCan, kvaser_send_float_array, ClearablePullConsumer
from eos.config import (
    PEDAL_SCALES,
    trucks_by_name,
    trucks_by_vin,
    Truck,
    can_servers_by_host,
    can_servers_by_name,
    trip_servers_by_name,
    trip_servers_by_host,
    generate_vcu_calibration,
)
from eos.utils import ragged_nparray_list_interp, GracefulKiller
from eos.visualization import plot_3d_figure, plot_to_image

# from bson import ObjectId


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


# local imports


# from utils import get_logger, get_truck_status, flash_vcu, plot_3d_figure
# value = [99.0] * 21 * 17
# send_float_array('TQD_trqTrqSetECO_MAP_v', value)

# system warnings and numpy warnings handling
warnings.filterwarnings("ignore", message="currentThread", category=DeprecationWarning)
np.warnings.filterwarnings("ignore", category=DeprecationWarning)


class RealtimeRDPG(object):
    def __init__(
        self,
        cloud=True,
        ui="cloud",
        resume=True,
        infer_mode=False,
        record=True,
        path=".",
        vehicle="HMZABAAH7MF011058",  # "VB7",
        driver="Longfei.Zheng",
        remotecan_srv="can_intra",
        web_srv="rocket_intra",
        mongo_srv="mongo_local",
        proj_root=Path("."),
        vlogger=None,
    ):
        self.cloud = cloud
        self.ui = ui
        self.trucks_by_name = trucks_by_name
        self.trucks_by_vin = trucks_by_vin
        self.vehicle = vehicle  # two possible values: "HMZABAAH7MF011058" or "VB7"
        self.remotecan_srv = remotecan_srv
        self.web_srv = web_srv
        self.mongo_srv = mongo_srv
        assert type(vehicle) == str, "vehicle must be a string"

        # Regex for VIN: HMZABAAH\wMF\d{6}
        p = re.compile(r"^HMZABAAH\wMF\d{6}$")
        if p.match(vehicle):
            # validate truck id
            # assert self.vehicle in self.trucks_by_vin.keys()
            self.truck = self.trucks_by_vin.get(self.vehicle)
            assert self.truck is not None, f"No Truck with VIN {self.vehicle}"
            self.truck_name = self.truck.TruckName  # 0: VB7, 1: VB6
        else:
            # validate truck id
            # assert self.vehicle in self.trucks_by_name.keys()
            self.truck = self.trucks_by_name.get(self.vehicle)
            assert self.truck is not None, f"No Truck with name {self.vehicle}"
            self.truck_name = self.truck.TruckName  # 0: VB7, 1: VB6
        self.driver = driver
        self.projroot = proj_root
        self.logger = vlogger
        self.dictLogger = dictLogger
        # self.dictLogger = {"user": inspect.currentframe().f_code.co_name}
        self.resume = resume
        self.infer_mode = infer_mode
        self.record = record
        self.path = path

        self.repo = Repo(self.projroot)
        # assert self.repo.is_dirty() == False, "Repo is dirty, please commit first"

        if resume:
            self.dataroot = projroot.joinpath(
                "data/" + self.truck.VIN + "−" + self.driver
            ).joinpath(self.path)
        else:
            self.dataroot = projroot.joinpath(
                "data/scratch/" + self.truck.VIN + "−" + self.driver
            ).joinpath(self.path)

        self.set_logger()
        self.logc.info(f"Start Logging", extra=self.dictLogger)
        self.logc.info(
            f"project root: {self.projroot}, git head: {str(self.repo.head.commit)[:7]}, author: {self.repo.head.commit.author}, git message: {self.repo.head.commit.message}",
            extra=self.dictLogger,
        )
        self.logc.info(f"vehicle: {self.vehicle}", extra=self.dictLogger)
        self.logc.info(f"driver: {self.driver}", extra=self.dictLogger)

        self.eps = np.finfo(
            np.float32
        ).eps.item()  # smallest number such that 1.0 + eps != 1.0

        if self.cloud:
            # reset proxy (internal site force no proxy)
            self.init_cloud()
            assert self.ui in [
                "cloud",
                "local",
                "mobile",
            ], f"ui must be cloud, local or mobile, not {self.ui}"
            if self.ui == "mobile":
                self.logger.info(f"Use phone UI", extra=self.dictLogger)
                self.get_truck_status = self.remote_webhmi_state_machine
            elif self.ui == "local":
                self.logger.info(f"Use local UI", extra=self.dictLogger)
                self.get_truck_status = self.remote_hmi_state_machine
            elif self.ui == "cloud":
                self.logger.info(f"Use cloud UI", extra=self.dictLogger)
                self.get_truck_status = self.remote_cloudhmi_state_machine
            else:
                raise ValueError("Unknown HMI type")
            self.flash_vcu = self.remote_flash_vcu
        else:
            self.get_truck_status = self.kvaser_get_truck_status
            self.flash_vcu = self.kvaser_flash_vcu

        self.logc.info(
            f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}"
        )
        gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
        tf.config.experimental.set_memory_growth(gpus[0], True)

        self.set_data_path()
        tf.keras.backend.set_floatx("float32")
        self.logc.info(
            f"tensorflow device lib:\n{device_lib.list_local_devices()}\n",
            extra=self.dictLogger,
        )
        self.logc.info(f"Tensorflow Imported!", extra=self.dictLogger)

        self.init_vehicle()
        self.build_actor_critic()
        self.logc.info(f"VCU and GPU Initialization done!", extra=self.dictLogger)
        self.init_threads_data()
        self.logc.info(f"Thread data Initialization done!", extra=self.dictLogger)

    def init_cloud(self):
        os.environ["http_proxy"] = ""
        self.can_server = can_servers_by_name.get(self.remotecan_srv)
        if self.can_server is None:
            self.can_server = can_servers_by_host.get(self.remotecan_srv.split(":")[0])
            assert (
                self.can_server is not None
            ), f"No such remotecan host {self.remotecan_srv} found!"
            assert (
                self.remotecan_srv.split(":")[1] == self.can_server.Port
            ), f"Port mismatch for remotecan host {self.remotecan_srv}!"
        self.logc.info(f"CAN Server found: {self.remotecan_srv}", extra=self.dictLogger)

        self.remotecan_client = RemoteCan(
            truckname=self.truck.TruckName,
            url="http://" + self.can_server.Host + ":" + self.can_server.Port + "/",
        )

        if self.ui == "mobile":
            self.trip_server = trip_servers_by_name.get(self.web_srv)
            if self.trip_server is None:
                self.trip_server = trip_servers_by_host.get(self.web_srv.split(":")[0])
                assert (
                    self.trip_server is not None
                ), f"No such trip server {self.web_srv} found!"
                assert (
                    self.web_srv.split(":")[1] == self.trip_server.Port
                ), f"Port mismatch for trip host {self.web_srv}!"
            self.logger.info(
                f"Trip Server found: {self.trip_server}", extra=self.dictLogger
            )

            # Create RocketMQ consumer
            self.rmq_consumer = ClearablePullConsumer("CID_EPI_ROCKET")
            self.rmq_consumer.set_namesrv_addr(
                self.trip_server.Host + ":" + self.trip_server.Port
            )

            # Create RocketMQ producer
            self.rmq_message_ready = Message("update_ready_state")
            self.rmq_message_ready.set_keys("what is keys mean")
            self.rmq_message_ready.set_tags("tags ------")
            self.rmq_message_ready.set_body(
                json.dumps({"vin": self.truck.VIN, "is_ready": True})
            )
            # self.rmq_message_ready.set_keys('trip_server')
            # self.rmq_message_ready.set_tags('tags')
            self.rmq_producer = Producer("PID-EPI_ROCKET")
            self.rmq_producer.set_namesrv_addr(
                self.trip_server.Host + ":" + self.trip_server.Port
            )

    def set_logger(self):
        self.logroot = self.dataroot.joinpath("py_logs")
        try:
            os.makedirs(self.logroot)
        except FileExistsError:
            print("User folder exists, just resume!")

        logfilename = self.logroot.joinpath(
            "eos-rt-rdpg-"
            + self.truck.TruckName
            + datetime.now().isoformat().replace(":", "-")
            + ".log"
        )
        formatter = logging.basicConfig(
            format="%(created)f-%(asctime)s.%(msecs)03d-%(name)s-%(levelname)s-%(module)s-%(threadName)s-%(funcName)s)-%(lineno)d): %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S.%f",
        )
        json_file_formatter = jsonlogger.JsonFormatter(
            "%(created)f %(asctime)s %(name)s %(levelname)s %(module)s %(threadName)s %(funcName)s) %(lineno)d) %(message)s"
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

        self.logc = logger.getChild("main")  # main thread control flow
        self.logc.propagate = True
        # self.logd = logger.getChild("data flow")
        # self.logd.propagate = True
        self.tflog = tf.get_logger()
        self.tflog.addHandler(fh)
        self.tflog.addHandler(ch)
        self.tflog.addHandler(skh)
        self.tflog.addHandler(strh)

        self.tableroot = self.dataroot.joinpath("tables")
        try:
            os.makedirs(self.tableroot)
        except FileExistsError:
            print("Table folder exists, just resume!")

    def set_data_path(self):
        # Create folder for ckpts loggings.
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = self.dataroot.joinpath(
            "tf_logs-vb/rdpg/gradient_tape/" + current_time + "/train"
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
            returncode, ret_str = self.remotecan_client.send_torque_map(
                pedalmap=self.vcu_calib_table1, swap=False
            )  # 14 rows for whole map
            self.logger.info(
                f"Done flash initial table. returncode: {returncode}, ret_str: {ret_str}",
                extra=self.dictLogger,
            )
        else:
            returncode = kvaser_send_float_array(self.vcu_calib_table1, sw_diff=False)
            self.logger.info(
                f"Done flash initial table. returncode: {returncode}",
                extra=self.dictLogger,
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
        self.state_len = self.observation_len * self.num_observations
        self.num_inputs = (
            self.num_observations * self.observation_len
        )  # 60 subsequent observations
        self.num_actions = self.vcu_calib_table_size  # 17*14 = 238
        self.vcu_calib_table_row_reduced = (
            self.truck.ActionFlashRow
        )  ## 0:5 adaptive rows correspond to low speed from  0~20, 7~25, 10~30, 15~35, etc  kmh  # overall action space is the whole table
        self.num_reduced_actions = (  # 0:4 adaptive rows correspond to low speed from  0~20, 7~30, 10~40, 20~50, etc  kmh  # overall action space is the whole table
            self.vcu_calib_table_row_reduced * self.vcu_calib_table_col
        )  # 4x17= 68
        # hyperparameters for DRL
        self.num_hidden = 256
        self.num_hidden0 = 16
        self.num_hidden1 = 32

        # DYNAMIC: need to adapt the pointer to change different roi of the pm, change the starting row index
        self.vcu_calib_table_row_start = 0

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

        self.h_t = []
        # Initialize networks
        self.rdpg = RDPG(
            self.truck,
            self.driver,
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
            cloud=self.cloud,
            db_server=self.mongo_srv,
            infer_mode=self.infer_mode,
        )

    # tracer.start()

    def init_threads_data(self):
        # multithreading initialization
        self.hmi_lock = Lock()
        self.state_machine_lock = Lock()
        self.tableQ_lock = Lock()
        self.captureQ_lock = Lock()
        self.remoteClient_lock = Lock()
        self.flash_env_lock = Lock()
        self.get_env_lock = Lock()
        self.done_env_lock = Lock()

        # tableQueue contains a table which is a list of type float
        self.tableQueue = queue.Queue()
        # motionpowerQueue contains a vcu states list with N(20) subsequent motion states + reward as observation
        self.motionpowerQueue = queue.Queue()

        # initial status of the switches
        self.program_exit = False
        self.program_start = False
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

    def capture_countdown_handler(
        self,
        evt_epi_done: threading.Event,
        evt_remote_get: threading.Event,
        evt_remote_flash: threading.Event,
    ):

        logger_countdown = self.logger.getChild("countdown")
        logger_countdown.propagate = True
        th_exit = False
        while not th_exit:
            with self.hmi_lock:
                if self.program_exit:
                    th_exit = True
                    continue

            logger_countdown.info(f"wait for countdown", extra=self.dictLogger)
            evt_epi_done.wait()
            with self.done_env_lock:
                evt_epi_done.clear()
            # if episode is done, sleep for the extension time
            time.sleep(self.epi_countdown_time)
            # cancel wait as soon as waking up
            logger_countdown.info(f"finish countdown", extra=self.dictLogger)

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

            # unlock remote_get_handler
            with self.get_env_lock:
                evt_remote_get.set()
            with self.flash_env_lock:
                evt_remote_flash.set()
            logger_countdown.info(
                f"Episode done! free remote_flash and remote_get!",
                extra=self.dictLogger,
            )
            if self.cloud is False:
                self.vel_hist_dQ.clear()
            # raise Exception("reset capture to stop")
        logger_countdown.info(f"Coutndown dies!!!", extra=self.dictLogger)

    def kvaser_get_truck_status(
        self,
        evt_epi_done: threading.Event,
        evt_remote_get: threading.Event,
        evt_remote_flash: threading.Event,
    ):
        """
        This function is used to get the truck status
        from the onboard udp socket server of CAN capture module Kvaser
        evt_remote_get is not used in this function for kvaser
        just to keep the uniform interface with remote_get_truck_status
        """

        th_exit = False
        logger_kvaser_get = self.logger.getChild("kvaser_get")
        logger_kvaser_get.propagate = True

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        socket.socket.settimeout(s, None)
        s.bind((self.get_truck_status_myHost, self.get_truck_status_myPort))
        # s.listen(5)
        logger_kvaser_get.info(f"Socket Initialization Done!", extra=self.dictLogger)

        self.vel_hist_dQ = deque(maxlen=20)  # accumulate 1s of velocity values
        # vel_cycle_dQ = deque(maxlen=30)  # accumulate 1.5s (one cycle) of velocity values
        vel_cycle_dQ = deque(
            maxlen=self.observation_len
        )  # accumulate 1.5s (one cycle) of velocity values

        while not th_exit:  # th_exit is local; program_exit is global
            with self.hmi_lock:  # wait for tester to kick off or to exit
                if self.program_exit == True:  # if program_exit is True, exit thread
                    logger_kvaser_get.info(
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
                logger_kvaser_get.critical(
                    f"udp sending wrong data type!", extra=self.dictLogger
                )
                raise TypeError("udp sending wrong data type!")

            for key, value in pop_data.items():
                if key == "status":  # state machine chores
                    # print(candata)
                    if value == "begin":
                        self.get_truck_status_start = True
                        logger_kvaser_get.info(
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
                        with self.done_env_lock:
                            evt_epi_done.set()
                        logger_kvaser_get.info(f"Episode end starts countdown!")
                        with self.hmi_lock:
                            # self.episode_count += 1  # valid round increments self.epi_countdown = False
                            self.episode_done = False  # TODO delay episode_done to make main thread keep running
                            self.episode_end = False
                    elif value == "end_invalid":
                        self.get_truck_status_start = False
                        logger_kvaser_get.info(
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
                        with self.done_env_lock:
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
                                    logger_kvaser_get.warning(
                                        f"cycle higher than 120km/h!",
                                        extra=self.dictLogger,
                                    )
                                    self.vcu_calib_table_row_start = 16
                                # get the row of the table

                                logger_kvaser_get.info(
                                    f"Cycle velocity: Aver{vel_aver:.2f},Min{vel_min:.2f},Max{vel_max:.2f},StartIndex{self.vcu_calib_table_row_start}!",
                                    extra=self.dictLogger,
                                )
                                # self.logc.info(
                                #     f"Producer Queue has {motionpowerQueue.qsize()}!", extra=self.dictLogger,
                                # )

                                with self.captureQ_lock:
                                    self.motionpowerQueue.put(
                                        self.get_truck_status_motpow_t
                                    )
                                self.get_truck_status_motpow_t = []
                    except Exception as X:
                        logger_kvaser_get.info(
                            X,  # f"Valid episode, Reset data capturing to stop after 3 seconds!",
                            extra=self.dictLogger,
                        )
                        break
                else:
                    logger_kvaser_get.warning(
                        f"udp sending message with key: {key}; value: {value}!!!"
                    )

                    break

        logger_kvaser_get.info(f"get_truck_status dies!!!", extra=self.dictLogger)

        s.close()

    # this is the calibration table consumer for flashing
    # @eye
    def kvaser_flash_vcu(self, evt_remote_flash: threading.Event):

        flash_count = 0
        th_exit = False

        logger_flash = self.logger.getChild("kvaser_flash")
        logger_flash.propagate = True

        logger_flash.info(f"Initialization Done!", extra=self.dictLogger)
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
                    curr_table_store_path = self.tableroot.joinpath(
                        "instant_table_rdpg-vb-"
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
                    logger_flash.info(
                        f"E{epi_cnt} done with record instant table: {step_count}",
                        extra=self.dictLogger,
                    )

                logger_flash.info(f"flash starts", extra=self.dictLogger)
                returncode = kvaser_send_float_array(
                    self.vcu_calib_table1, sw_diff=True
                )
                # time.sleep(1.0)

                if returncode != 0:
                    logger_flash.error(
                        f"kvaser_send_float_array failed: {returncode}",
                        extra=self.dictLogger,
                    )
                else:
                    logger_flash.info(
                        f"flash done, count:{flash_count}", extra=self.dictLogger
                    )
                    flash_count += 1
                # watch(flash_count)

        logger_flash.info(f"Save the last table!!!!", extra=self.dictLogger)
        last_table_store_path = (
            self.dataroot.joinpath(  #  there's no slash in the end of the string
                "last_table_ddpg-"
                + datetime.now().strftime("%y-%m-%d-%H-%M-%S")
                + ".csv"
            )
        )
        with open(last_table_store_path, "wb") as f:
            self.vcu_calib_table1.to_csv(last_table_store_path)
        logger_flash.info(f"flash_vcu dies!!!", extra=self.dictLogger)

    def remote_get_handler(
        self, evt_remote_get: threading.Event, evt_remote_flash: threading.Event
    ):

        th_exit = False
        logger_remote_get = self.logger.getChild("remote_get")
        logger_remote_get.propagate = True

        while not th_exit:
            with self.hmi_lock:
                if self.program_exit:
                    th_exit = self.program_exit
                    continue
                episode_end = self.episode_end
            if episode_end is True:
                logger_remote_get.info(
                    f"Episode ends and wait for evt_remote_get!",
                    extra=self.dictLogger,
                )
                with self.get_env_lock:
                    evt_remote_get.clear()
                # continue

            logger_remote_get.info(
                f"wait for remote get trigger", extra=self.dictLogger
            )
            evt_remote_get.wait()

            # after long wait, need to refresh state machine
            with self.hmi_lock:
                th_exit = self.program_exit
                episode_end = self.episode_end

            if episode_end is True:
                logger_remote_get.info(
                    f"Episode ends after evt_remote_get without get_signals!",
                    extra=self.dictLogger,
                )
                with self.get_env_lock:
                    evt_remote_get.clear()
                continue

            # if episode is done, sleep for the extension time
            # cancel wait as soon as waking up
            timeout = self.truck.CloudUnitNumber + 7
            logger_remote_get.info(
                f"Wake up to fetch remote data, duration={self.truck.CloudUnitNumber}s timeout={timeout}s",
                extra=self.dictLogger,
            )
            with self.remoteClient_lock:
                (signal_success, remotecan_data,) = self.remotecan_client.get_signals(
                    duration=self.truck.CloudUnitNumber, timeout=timeout
                )  # timeout is 1 second longer than duration
                if signal_success != 0:  # in case of failure, ping server
                    logger_remote_get.warning(
                        f"RemoteCAN failure! return state={signal_success}s, return_code={remotecan_data}",
                        extra=self.dictLogger,
                    )

                    response = os.system("ping -c 1 " + self.can_server.Host)
                    if response == 0:
                        logger_remote_get.info(
                            f"{self.can_server.Host} is up!", extra=self.dictLogger
                        )
                    else:
                        logger_remote_get.info(
                            f"{self.can_server.Host} is down!", extra=self.dictLogger
                        )
                    # ping test
                    # try:
                    #     response_ping = subprocess.check_output(
                    #         "ping -c 1 " + self.can_server.Host, shell=True
                    #     )
                    # except subprocess.CalledProcessError as e:
                    #     logger_remote_get.info(
                    #         f"{self.can_server.Host} is down, responds: {response_ping}"
                    #         f"return code: {e.returncode}, output: {e.output}!",
                    #         extra=self.dictLogger,
                    #     )
                    # logger_remote_get.info(
                    #     f"{self.can_server.Host} is up, responds: {response_ping}!",
                    #     extra=self.dictLogger,
                    # )
                    #
                    # # telnet test
                    # try:
                    #     response_telnet = subprocess.check_output(
                    #         f"timeout 1 telnet {self.can_server.Host} {self.can_server.Port}",
                    #         shell=True,
                    #     )
                    #     logger_remote_get.info(
                    #         f"Telnet {self.can_server.Host} responds: {response_telnet}!",
                    #         extra=self.dictLogger,
                    #     )
                    # except subprocess.CalledProcessError as e:
                    #     logger_remote_get.info(
                    #         f"telnet {self.can_server.Host} return code: {e.returncode}, output: {e.output}!",
                    #         extra=self.dictLogger,
                    #     )
                    # except subprocess.TimeoutExpired as e:
                    #     logger_remote_get.info(
                    #         f"telnet {self.can_server.Host} timeout"
                    #         f"cmd: {e.cmd}, output: {e.output}, timeout: {e.timeout}!",
                    #         extra=self.dictLogger,
                    #     )

            if not isinstance(remotecan_data, dict):
                logger_remote_get.critical(
                    f"udp sending wrong data type!",
                    extra=self.dictLogger,
                )
                raise TypeError("udp sending wrong data type!")
            else:
                logger_remote_get.info(
                    f"Get remote data, signal_success={signal_success}!",
                    extra=self.dictLogger,
                )

            try:
                if signal_success == 0:

                    with self.hmi_lock:
                        th_exit = self.program_exit
                        episode_end = self.episode_end
                    if episode_end is True:
                        logger_remote_get.info(
                            f"Episode ends, not waiting for evt_remote_flash and continue!",
                            extra=self.dictLogger,
                        )
                        with self.get_env_lock:
                            evt_remote_get.clear()
                        continue

                    try:
                        signal_freq = self.truck.CloudSignalFrequency
                        gear_freq = self.truck.CloudGearFrequency
                        unit_duration = self.truck.CloudUnitDuration
                        unit_ob_num = unit_duration * signal_freq
                        unit_gear_num = unit_duration * gear_freq
                        unit_num = self.truck.CloudUnitNumber
                        for key, value in remotecan_data.items():
                            if key == "result":
                                logger_remote_get.info(
                                    "convert observation state to array.",
                                    extra=self.dictLogger,
                                )
                                # timestamp processing
                                timestamps = []
                                separators = "--T::."  # adaption separators of the raw intest string
                                start_century = "20"
                                for ts in value["timestamps"]:
                                    # create standard iso string datetime format
                                    ts_substrings = [
                                        ts[i : i + 2] for i in range(0, len(ts), 2)
                                    ]
                                    ts_iso = start_century
                                    for i, sep in enumerate(separators):
                                        ts_iso = ts_iso + ts_substrings[i] + sep
                                    ts_iso = ts_iso + ts_substrings[-1]
                                    timestamps.append(ts_iso)
                                timestamps_units = (
                                    np.array(timestamps).astype("datetime64[ms]")
                                    - np.timedelta64(8, "h")
                                ).astype(  # convert to UTC+8
                                    "int"
                                )  # convert to int
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
                                    logger_remote_get.warning(
                                        f"cycle higher than 120km/h!",
                                        extra=self.dictLogger,
                                    )
                                    self.vcu_calib_table_row_start = 16

                                logger_remote_get.info(
                                    f"Cycle velocity: Aver{np.mean(velocity):.2f},Min{np.amin(velocity):.2f},Max{np.amax(velocity):.2f},StartIndex{self.vcu_calib_table_row_start}!",
                                    extra=self.dictLogger,
                                )

                                with self.captureQ_lock:
                                    self.motionpowerQueue.put(motion_power)

                                logger_remote_get.info(
                                    f"Get one record, wait for remote_flash!!!",
                                    extra=self.dictLogger,
                                )
                                # as long as one observation is received, always waiting for flash
                                evt_remote_flash.wait()
                                with self.flash_env_lock:
                                    evt_remote_flash.clear()
                                logger_remote_get.info(
                                    f"evt_remote_flash wakes up, reset inner lock, restart remote_get!!!",
                                    extra=self.dictLogger,
                                )
                            else:
                                # self.logger.info(
                                #     f"show status: {key}:{value}",
                                #     extra=self.dictLogger,
                                # )
                                pass
                    except Exception as X:
                        logger_remote_get.error(
                            f"Observation Corrupt! Status exception {X}",
                            extra=self.dictLogger,
                        )
                else:
                    logger_remote_get.error(
                        f"get_signals failed: {remotecan_data}",
                        extra=self.dictLogger,
                    )

            except Exception as X:
                logger_remote_get.info(
                    f"Break due to Exception: {X}",
                    extra=self.dictLogger,
                )

            with self.get_env_lock:
                evt_remote_get.clear()

        logger_remote_get.info(f"thr_remoteget dies!!!!!", extra=self.dictLogger)

    def remote_webhmi_state_machine(
        self,
        evt_epi_done: threading.Event,
        evt_remote_get: threading.Event,
        evt_remote_flash: threading.Event,
    ):
        """
        This function is used to get the truck status
        from remote can module
        """
        logger_webhmi_sm = self.logger.getChild("webhmi_sm")
        logger_webhmi_sm.propagate = True
        th_exit = False

        try:
            self.rmq_consumer.start()
            self.rmq_producer.start()
            logger_webhmi_sm.info(
                f"Start RocketMQ client on {self.trip_server.Host}!",
                extra=self.dictLogger,
            )

            msg_topic = self.driver + "_" + self.truck.VIN

            broker_msgs = self.rmq_consumer.pull(msg_topic)
            logger_webhmi_sm.info(
                f"Before clearing history: Pull {len(list(broker_msgs))} history messages of {msg_topic}!",
                extra=self.dictLogger,
            )
            self.rmq_consumer.clear_history(msg_topic)
            broker_msgs = self.rmq_consumer.pull(msg_topic)
            logger_webhmi_sm.info(
                f"After clearing history: Pull {len(list(broker_msgs))} history messages of {msg_topic}!",
                extra=self.dictLogger,
            )
            all(broker_msgs)  # exhaust history messages

        except Exception as e:
            logger_webhmi_sm.error(
                f"send_sync failed: {e}",
                extra=self.dictLogger,
            )
            return
        try:
            # send ready signal to trip server
            ret = self.rmq_producer.send_sync(self.rmq_message_ready)
            logger_webhmi_sm.info(
                f"Sending ready signal to trip server:"
                f"status={ret.status};"
                f"msg-id={ret.msg_id};"
                f"offset={ret.offset}.",
                extra=self.dictLogger,
            )
            with self.state_machine_lock:
                self.program_start = True

            logger_webhmi_sm.info(
                f"RocketMQ client Initialization Done!", extra=self.dictLogger
            )
        except Exception as e:
            logger_webhmi_sm.error(
                f"Fatal Failure!: {e}",
                extra=self.dictLogger,
            )
            return

        while not th_exit:  # th_exit is local; program_exit is global
            with self.hmi_lock:  # wait for tester to kick off or to exit
                if self.program_exit == True:  # if program_exit is True, exit thread
                    logger_webhmi_sm.info(
                        "%s",
                        "Capture thread exit due to processing request!!!",
                        extra=self.dictLogger,
                    )
                    th_exit = True
                    continue
            msgs = self.rmq_consumer.pull(msg_topic)
            for msg in msgs:
                msg_body = json.loads(msg.body)
                if not isinstance(msg_body, dict):
                    logger_webhmi_sm.critical(
                        f"rocketmq server sending wrong data type!",
                        extra=self.dictLogger,
                    )
                    raise TypeError("rocketmq server sending wrong data type!")
                logger_webhmi_sm.info(f"Get message {msg_body}!", extra=self.dictLogger)
                if msg_body["vin"] != self.truck.VIN:
                    continue

                if msg_body["code"] == 5:  # "config/start testing"
                    logger_webhmi_sm.info(
                        f"Restart/Reconfigure message VIN: {msg_body['vin']}; driver {msg_body['name']}!",
                        extra=self.dictLogger,
                    )

                    with self.state_machine_lock:
                        self.program_start = True

                    # send ready signal to trip server
                    ret = self.rmq_producer.send_sync(self.rmq_message_ready)
                    logger_webhmi_sm.info(
                        f"Sending ready signal to trip server:"
                        f"status={ret.status};"
                        f"msg-id={ret.msg_id};"
                        f"offset={ret.offset}.",
                        extra=self.dictLogger,
                    )
                elif msg_body["code"] == 1:  # start episode

                    self.get_truck_status_start = True
                    logger_webhmi_sm.info(
                        "%s", "Episode will start!!!", extra=self.dictLogger
                    )
                    th_exit = False
                    # ts_epi_start = time.time()
                    with self.get_env_lock:
                        evt_remote_get.clear()
                    with self.flash_env_lock:
                        evt_remote_flash.clear()
                    logger_webhmi_sm.info(
                        f"Episode start! clear remote_flash and remote_get!",
                        extra=self.dictLogger,
                    )

                    with self.captureQ_lock:
                        while not self.motionpowerQueue.empty():
                            self.motionpowerQueue.get()
                    with self.hmi_lock:
                        self.episode_done = False
                        self.episode_end = False
                elif msg_body["code"] == 2:  # valid stop

                    # DONE for valid end wait for another 2 queue objects (3 seconds) to get the last reward!
                    # cannot sleep the thread since data capturing in the same thread, use signal alarm instead

                    logger_webhmi_sm.info("End Valid!!!!!!", extra=self.dictLogger)
                    self.get_truck_status_start = (
                        True  # do not stopping data capture immediately
                    )

                    # set flag for countdown thread
                    with self.done_env_lock:
                        evt_epi_done.set()

                    logger_webhmi_sm.info(f"Episode end starts countdown!")
                    with self.hmi_lock:
                        # self.episode_count += 1  # valid round increments self.epi_countdown = False
                        self.episode_done = False  # TODO delay episode_done to make main thread keep running
                        self.episode_end = False
                elif msg_body["code"] == 3:  # invalid stop

                    self.get_truck_status_start = False
                    logger_webhmi_sm.info(
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

                    # remote_get_handler exit
                    with self.get_env_lock:
                        evt_remote_get.set()
                    with self.flash_env_lock:
                        evt_remote_flash.set()
                    logger_webhmi_sm.info(
                        f"end_invalid! free remote_flash and remote_get!",
                        extra=self.dictLogger,
                    )

                    with self.hmi_lock:
                        self.episode_done = False
                        self.episode_end = True
                        self.episode_count += 1  # invalid round increments
                elif msg_body["code"] == 4:  # "exit"
                    self.get_truck_status_start = False
                    self.get_truck_status_motpow_t = []

                    with self.get_env_lock:
                        evt_remote_get.set()
                    with self.flash_env_lock:
                        evt_remote_flash.set()
                    logger_webhmi_sm.info(
                        f"Program exit!!!! free remote_flash and remote_get!",
                        extra=self.dictLogger,
                    )

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
                    with self.done_env_lock:
                        evt_epi_done.set()
                    break
                    # time.sleep(0.1)
                else:
                    logger_webhmi_sm.warning(
                        f"Unknown message {msg_body}!", extra=self.dictLogger
                    )

            time.sleep(0.05)  # sleep for 50ms to update state machine
            if self.get_truck_status_start:
                with self.get_env_lock:
                    evt_remote_get.set()

        self.rmq_consumer.shutdown()
        self.rmq_producer.shutdown()
        logger_webhmi_sm.info(f"remote webhmi dies!!!", extra=self.dictLogger)

    def remote_cloudhmi_state_machine(
        self,
        evt_epi_done: threading.Event,
        evt_remote_get: threading.Event,
        evt_remote_flash: threading.Event,
    ):
        """
        This function is used to get the truck status
        from cloud state management and remote can module
        """

        th_exit = False

        logger_cloudhmi_sm = self.logger.getChild("cloudhmi_sm")
        logger_cloudhmi_sm.propagate = True

        logger_cloudhmi_sm.info(
            f"Start/Configure message VIN: {self.truck.VIN}; driver {self.driver}!",
            extra=self.dictLogger,
        )

        with self.state_machine_lock:
            self.program_start = True

        logger_cloudhmi_sm.info(
            "%s",
            "Road Test with inferring will start as one single episode!!!",
            extra=self.dictLogger,
        )
        while not th_exit:  # th_exit is local; program_exit is global

            with self.hmi_lock:  # wait for tester to kick off or to exit
                # Check if the runner is trying to kill the process
                # kill signal captured from main thread
                if self.program_exit == True:  # if program_exit is True, exit thread
                    logger_cloudhmi_sm.info(
                        "%s",
                        "UI thread exit due to processing request!!!",
                        extra=self.dictLogger,
                    )

                    self.get_truck_status_start = False
                    self.get_truck_status_motpow_t = []

                    with self.get_env_lock:
                        evt_remote_get.set()
                    with self.flash_env_lock:
                        evt_remote_flash.set()

                    logger_cloudhmi_sm.info(
                        f"Process is being killed and Program exit!!!! free remote_flash and remote_get!",
                        extra=self.dictLogger,
                    )

                    with self.captureQ_lock:
                        while not self.motionpowerQueue.empty():
                            self.motionpowerQueue.get()

                    self.episode_done = True
                    self.episode_end = True
                    self.episode_count += 1

                    with self.done_env_lock:
                        evt_epi_done.set()
                    th_exit = True
                    continue

            # ts_epi_start = time.time()
            with self.get_env_lock:
                evt_remote_get.clear()
            with self.flash_env_lock:
                evt_remote_flash.clear()
            # logger_cloudhmi_sm.info(
            #     f"Test start! clear remote_flash and remote_get!",
            #     extra=self.dictLogger,
            # )

            with self.captureQ_lock:
                while not self.motionpowerQueue.empty():
                    self.motionpowerQueue.get()
            with self.hmi_lock:
                self.episode_done = False
                self.episode_end = False

            time.sleep(0.05)  # sleep for 50ms to update state machine
            with self.get_env_lock:
                evt_remote_get.set()

        logger_cloudhmi_sm.info(
            f"remote cloudhmi killed gracefully!!!", extra=self.dictLogger
        )

    def remote_hmi_state_machine(
        self,
        evt_epi_done: threading.Event,
        evt_remote_get: threading.Event,
        evt_remote_flash: threading.Event,
    ):
        """
        This function is used to get the truck status
        from remote can module
        """

        th_exit = False

        logger_hmi_sm = self.logger.getChild("hmi_sm")
        logger_hmi_sm.propagate = True
        #  Get the HMI control command from UDP, but not the data from KvaserCAN
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        socket.socket.settimeout(s, None)
        s.bind((self.get_truck_status_myHost, self.get_truck_status_myPort))
        # s.listen(5)
        logger_hmi_sm.info(f"Socket Initialization Done!", extra=self.dictLogger)

        while not th_exit:  # th_exit is local; program_exit is global
            with self.hmi_lock:  # wait for tester to kick off or to exit
                if self.program_exit == True:  # if program_exit is True, exit thread
                    logger_hmi_sm.info(
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
                logger_hmi_sm.critical(
                    f"udp sending wrong data type!", extra=self.dictLogger
                )
                raise TypeError("udp sending wrong data type!")

            for key, value in pop_data.items():
                if key == "status":  # state machine chores
                    # print(candata)
                    # self.logc.info(
                    #     f"Status data: key={key},value={value}!!!!!!", extra=self.dictLogger
                    # )
                    if value == "begin":
                        self.get_truck_status_start = True
                        logger_hmi_sm.info(
                            "%s", "Episode will start!!!", extra=self.dictLogger
                        )
                        th_exit = False
                        # ts_epi_start = time.time()
                        with self.get_env_lock:
                            evt_remote_get.clear()
                        with self.flash_env_lock:
                            evt_remote_flash.clear()
                        logger_hmi_sm.info(
                            f"Episode start! clear remote_flash and remote_get!",
                            extra=self.dictLogger,
                        )

                        with self.captureQ_lock:
                            while not self.motionpowerQueue.empty():
                                self.motionpowerQueue.get()
                        with self.hmi_lock:
                            self.episode_done = False
                            self.episode_end = False
                    elif value == "end_valid":
                        # DONE for valid end wait for another 2 queue objects (3 seconds) to get the last reward!
                        # cannot sleep the thread since data capturing in the same thread, use signal alarm instead

                        logger_hmi_sm.info("End Valid!!!!!!", extra=self.dictLogger)
                        self.get_truck_status_start = (
                            True  # do not stopping data capture immediately
                        )

                        # set flag for countdown thread
                        with self.done_env_lock:
                            evt_epi_done.set()
                        logger_hmi_sm.info(f"Episode end starts countdown!")
                        with self.hmi_lock:
                            # self.episode_count += 1  # valid round increments self.epi_countdown = False
                            self.episode_done = False  # TODO delay episode_done to make main thread keep running
                            self.episode_end = False
                    elif value == "end_invalid":
                        self.get_truck_status_start = False
                        logger_hmi_sm.info(
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

                        # remote_get_handler exit
                        with self.get_env_lock:
                            evt_remote_get.set()
                        with self.flash_env_lock:
                            evt_remote_flash.set()
                        logger_hmi_sm.info(
                            f"end_invalid! free remote_flash and remote_get!",
                            extra=self.dictLogger,
                        )

                        with self.hmi_lock:
                            self.episode_done = False
                            self.episode_end = True
                            self.episode_count += 1  # invalid round increments
                    elif value == "exit":
                        self.get_truck_status_start = False
                        self.get_truck_status_motpow_t = []

                        with self.get_env_lock:
                            evt_remote_get.set()
                        with self.flash_env_lock:
                            evt_remote_flash.set()
                        logger_hmi_sm.info(
                            f"Program exit!!!! free remote_flash and remote_get!",
                            extra=self.dictLogger,
                        )

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
                        with self.done_env_lock:
                            evt_epi_done.set()
                        break
                        # time.sleep(0.1)
                elif key == "data":
                    #  instead of get kvasercan, we get remotecan data here!
                    if self.get_truck_status_start:  # starts episode
                        # set flag for remote_get thread
                        with self.get_env_lock:
                            evt_remote_get.set()
                        # self.logc.info(f"Kick off remoteget!!")
                else:
                    logger_hmi_sm.warning(
                        f"udp sending message with key: {key}; value: {value}!!!"
                    )

                    break

        s.close()
        logger_hmi_sm.info(f"remote hmi dies!!!", extra=self.dictLogger)

    def remote_flash_vcu(self, evt_remote_flash: threading.Event):
        """
        trigger 1: tableQueue is not empty
        trigger 2: remote client is available as signaled by the remote_get thread
        """
        flash_count = 0
        th_exit = False

        logger_flash = self.logger.getChild("flash")
        logger_flash.propagate = True
        logger_flash.info(f"Initialization Done!", extra=self.dictLogger)
        while not th_exit:
            # time.sleep(0.1)

            with self.state_machine_lock:
                program_start = self.program_start
            if program_start is False:
                continue

            with self.hmi_lock:
                table_start = self.vcu_calib_table_row_start
                epi_cnt = self.episode_count
                step_count = self.step_count
                if self.program_exit:
                    th_exit = True
                    continue

            # self.logc.info(f"Wait for table!", extra=self.dictLogger)
            try:
                # print("1 tablequeue size: {}".format(tablequeue.qsize()))
                with self.tableQ_lock:
                    table = self.tableQueue.get(
                        block=False, timeout=1
                    )  # default block = True
                    # print("2 tablequeue size: {}".format(tablequeue.qsize()))

                with self.hmi_lock:
                    th_exit = self.program_exit
                    episode_end = self.episode_end

                if episode_end is True:
                    with self.flash_env_lock:
                        evt_remote_flash.set()  # triggered flash by remote_get thread, need to reset remote_get waiting evt
                    logger_flash.info(
                        f"Episode ends, skipping remote_flash and continue!",
                        extra=self.dictLogger,
                    )
                    continue
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

                    curr_table_store_path = self.tableroot.joinpath(
                        "instant_table_ddpg-vb-"
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
                    # self.logc.info(
                    #     f"E{epi_cnt} done with record instant table: {step_count}",
                    #     extra=self.dictLogger,
                    # )

                # with self.hmi_lock:
                #     th_exit = self.program_exit
                #     episode_end = self.episode_end
                #
                # if episode_end is True:
                #     evt_remote_flash.set()  # triggered flash by remote_get thread, need to reset remote_get waiting evt
                #     self.logc.info(
                #         f"Episode ends, skipping remote_flash and continue!",
                #         extra=self.dictLogger,
                #     )
                #     continue

                # empirically, 1s is enough for 1 row, 4 rows need 5 seconds
                timeout = self.vcu_calib_table_row_reduced + 3
                logger_flash.info(
                    f"flash starts, timeout={timeout}s", extra=self.dictLogger
                )
                # lock doesn't control the logic explictitly
                # competetion is not desired
                with self.remoteClient_lock:
                    returncode, ret_str = self.remotecan_client.send_torque_map(
                        pedalmap=self.vcu_calib_table1.iloc[
                            table_start : self.vcu_calib_table_row_reduced + table_start
                        ],
                        swap=False,
                        timeout=timeout,
                    )
                # time.sleep(1.0)
                if returncode != 0:
                    logger_flash.error(
                        f"send_torque_map failed and retry: {returncode}, ret_str: {ret_str}",
                        extra=self.dictLogger,
                    )

                    response = os.system("ping -c 1 " + self.can_server.Url)
                    if response == 0:
                        logger_flash.info(
                            f"{self.can_server.Url} is up!", extra=self.dictLogger
                        )
                    else:
                        logger_flash.info(
                            f"{self.can_server.Url} is down!", extra=self.dictLogger
                        )
                else:
                    logger_flash.info(
                        f"flash done, count:{flash_count}", extra=self.dictLogger
                    )
                    flash_count += 1

                # flash is done and unlock remote_get
                with self.flash_env_lock:
                    evt_remote_flash.set()

                # watch(flash_count)

        logger_flash.info(f"Save the last table!!!!", extra=self.dictLogger)

        last_table_store_path = (
            self.dataroot.joinpath(  #  there's no slash in the end of the string
                "last_table_ddpg-"
                + datetime.now().strftime("%y-%m-%d-%H-%M-%S")
                + ".csv"
            )
        )
        with open(last_table_store_path, "wb") as f:
            self.vcu_calib_table1.to_csv(last_table_store_path)
        # motionpowerQueue.join()
        logger_flash.info(f"remote_flash_vcu dies!!!", extra=self.dictLogger)

    # @eye
    def run(self):

        # Start thread for flashing vcu, flash first
        evt_epi_done = threading.Event()
        evt_remote_get = threading.Event()
        evt_remote_flash = threading.Event()
        self.thr_countdown = Thread(
            target=self.capture_countdown_handler,
            name="countdown",
            args=[evt_epi_done, evt_remote_get, evt_remote_flash],
        )
        self.thr_countdown.start()

        self.thr_observe = Thread(
            target=self.get_truck_status,
            name="observe",
            args=[evt_epi_done, evt_remote_get, evt_remote_flash],
        )
        self.thr_observe.start()

        if self.cloud:
            self.thr_remoteget = Thread(
                target=self.remote_get_handler,
                name="remoteget",
                args=[evt_remote_get, evt_remote_flash],
            )
            self.thr_remoteget.start()

        self.thr_flash = Thread(
            target=self.flash_vcu, name="flash", args=[evt_remote_flash]
        )
        self.thr_flash.start()

        """
        ## train
        """
        running_reward = 0
        th_exit = False
        epi_cnt_local = 0

        # Gracefulkiller only in the main thread!
        killer = GracefulKiller()

        self.logc.info(f"main Initialization done!", extra=self.dictLogger)
        while not th_exit:  # run until solved or program exit; th_exit is local
            with self.hmi_lock:  # wait for tester to kick off or to exit
                th_exit = self.program_exit  # if program_exit is False,
                epi_cnt = self.episode_count  # get episode counts
                epi_end = self.episode_end

            with self.state_machine_lock:
                program_start = self.program_start
            if program_start is False:
                continue

            if epi_end:  # if episode_end is True, wait for start of episode
                # self.logger.info(f'wait for start!', extra=self.dictLogger)
                continue

            step_count = 0
            episode_reward = 0
            # tf.summary.trace_on(graph=True, profiler=True)

            self.logc.info("----------------------", extra=self.dictLogger)
            self.logc.info(
                f"E{epi_cnt} starts!",
                extra=self.dictLogger,
            )

            self.rdpg.start_episode(datetime.now(tz=self.truck.tz))
            tf.debugging.set_log_device_placement(True)
            with tf.device("/GPU:0"):
                while (
                    not epi_end
                ):  # end signal, either the round ends normally or user interrupt
                    if killer.kill_now:
                        self.logc.info(f"Process is being killed!!!")
                        with self.hmi_lock:
                            self.program_exit = True

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
                    # self.logc.info(f"motionpowerQueue.qsize(): {motionpowerqueue_size}")
                    if epi_end and done and (motionpowerqueue_size > 2):
                        # self.logc.info(f"motionpowerQueue.qsize(): {self.motionpowerQueue.qsize()}")
                        self.logc.info(
                            f"Residue in Queue is a sign of disordered sequence, interrupted!"
                        )
                        done = (
                            False  # this local done is true done with data exploitation
                        )

                    if epi_end:  # stop observing and inferring
                        continue

                    try:
                        # self.logc.info(
                        #     f"E{epi_cnt} Wait for an object!!!", extra=self.dictLogger
                        # )

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
                    motpow_t = tf.convert_to_tensor(
                        motionpower
                    )  # state must have 30 (velocity, pedal, brake, current, voltage) 5 tuple (num_observations)
                    if self.cloud:
                        out = tf.split(
                            motpow_t, [1, 3, 1, 2], 1
                        )  # note the difference of split between np and tf
                        (ts, o_t0, gr_t, pow_t) = [tf.squeeze(x) for x in out]
                        o_t = tf.reshape(o_t0, -1)
                    else:
                        o_t0, pow_t = tf.split(motpow_t, [3, 2], 1)
                        o_t = tf.reshape(o_t0, -1)

                    self.logc.info(
                        f"E{epi_cnt} tensor convert and split!",
                        extra=self.dictLogger,
                    )
                    ui_sum = tf.reduce_sum(
                        tf.reduce_prod(pow_t, 1)
                    )  # vcu reward is a scalar
                    wh = (
                        ui_sum / 3600.0 * self.sample_rate
                    )  # rate 0.05 for kvaser, 0.02 remote # negative wh
                    # self.logger.info(
                    #     f"ui_sum: {ui_sum}",
                    #     extra=self.dictLogger,
                    # )
                    self.logc.info(
                        f"wh: {wh}",
                        extra=self.dictLogger,
                    )

                    # !!!no parallel even!!!
                    # predict action probabilities and estimated future rewards
                    # from environment state
                    # for causal rl, the odd indexed observation/reward are caused by last action
                    # skip the odd indexed observation/reward for policy to make it causal
                    self.logc.info(
                        f"E{epi_cnt} before inference!",
                        extra=self.dictLogger,
                    )
                    # motion states o_t is 30*3/50*3 matrix
                    a_t = self.rdpg.actor_predict(o_t, int(step_count / 1))
                    # self.logc.info(
                    #     f"E{epi_cnt} step{int(step_count/2)},o_t.shape:{o_t.shape},a_t.shape:{a_t.shape}!",
                    #     extra=self.dictLogger,
                    # )
                    self.logc.info(
                        f"E{epi_cnt} inference done with reduced action space!",
                        extra=self.dictLogger,
                    )

                    with self.tableQ_lock:
                        self.tableQueue.put(a_t)
                        self.logc.info(
                            f"E{epi_cnt} StartIndex {table_start} Action Push table: {self.tableQueue.qsize()}",
                            extra=self.dictLogger,
                        )
                    self.logc.info(
                        f"E{epi_cnt} Finish Inference Step: {step_count}",
                        extra=self.dictLogger,
                    )

                    # !!!no parallel even!!!
                    cycle_reward = wh * (-1.0)
                    episode_reward += cycle_reward

                    if step_count > 0:
                        self.rdpg.deposit(
                            prev_o_t, prev_a_t, prev_table_start, cycle_reward
                        )

                    prev_o_t = o_t
                    prev_a_t = a_t
                    prev_table_start = table_start

                    # TODO add speed sum as positive reward
                    self.logc.info(
                        f"E{epi_cnt} Step done: {step_count}",
                        extra=self.dictLogger,
                    )

                    # during odd steps, old action remains effective due to learn and flash delay
                    # so ust record the reward history
                    # motion states (observation) are not used later for backpropagation

                    # step level
                    step_count += 1

            if (
                not done
            ):  # if user interrupt prematurely or exit, then ignore back propagation since data incomplete
                self.logc.info(
                    f"E{epi_cnt} interrupted, waits for next episode to kick off!",
                    extra=self.dictLogger,
                )
                # send ready signal to trip server
                if self.ui == "mobile":
                    ret = self.rmq_producer.send_sync(self.rmq_message_ready)
                    self.logc.info(
                        f"Sending ready signal to trip server:"
                        f"status={ret.status};"
                        f"msg-id={ret.msg_id};"
                        f"offset={ret.offset}.",
                        extra=self.dictLogger,
                    )
                continue  # otherwise assuming the history is valid and back propagate

            self.rdpg.end_episode()  # deposit history

            self.logc.info(
                f"E{epi_cnt} Experience Collection ends!",
                extra=self.dictLogger,
            )

            critic_loss = 0
            actor_loss = 0
            if self.infer_mode:
                # FIXME bugs in maximal sequence length for ungraceful testing
                # (actor_loss, critic_loss) = self.rdpg.notrain()
                self.logc.info("No Learning, just calculating loss")

            else:
                self.logc.info("Learning and soft updating 6 times")
                for k in range(6):
                    # self.logger.info(f"BP{k} starts.", extra=self.dictLogger)
                    if self.rdpg.buffer_counter > 0:
                        (actor_loss, critic_loss) = self.rdpg.train()
                        self.rdpg.soft_update_target()

                    else:
                        self.logc.info(
                            f"Buffer empty, no learning!", extra=self.dictLogger
                        )
                        self.logc.info(
                            "++++++++++++++++++++++++", extra=self.dictLogger
                        )
                        break

                # Checkpoint manager save model
                self.rdpg.save_ckpt()

            self.logc.info(
                f"E{epi_cnt}BP 6 times critic loss: {critic_loss}; actor loss: {actor_loss}",
                extra=self.dictLogger,
            )

            # update running reward to check condition for solving
            running_reward = 0.05 * (-episode_reward) + (1 - 0.05) * running_reward

            # Create a matplotlib 3d figure, //export and save in log
            fig = plot_3d_figure(self.vcu_calib_table1)

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
                    "Calibration Table Hist",
                    self.vcu_calib_table1.to_numpy().tolist(),
                    step=epi_cnt_local,
                )
                # tf.summary.trace_export(
                #     name="veos_trace", step=epi_cnt_local, profiler_outdir=self.train_log_dir
                # )

            epi_cnt_local += 1
            plt.close(fig)

            self.logc.info(
                f"E{epi_cnt} Episode Reward: {episode_reward}",
                extra=self.dictLogger,
            )

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

            # send ready signal to trip server
            if self.ui == "mobile":
                ret = self.rmq_producer.send_sync(self.rmq_message_ready)
                self.logger.info(
                    f"Sending ready signal to trip server:"
                    f"status={ret.status};"
                    f"msg-id={ret.msg_id};"
                    f"offset={ret.offset}.",
                    extra=self.dictLogger,
                )
        # TODO terminate condition to be defined: reward > limit (percentage); time too long
        # with self.train_summary_writer.as_default():
        #     tf.summary.trace_export(
        #         name="veos_trace",
        #         step=epi_cnt_local,
        #         profiler_outdir=self.train_log_dir,
        #     )
        self.thr_observe.join()
        if self.cloud:
            self.thr_remoteget.join()
        self.thr_flash.join()
        self.thr_countdown.join()

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
        default=True,
        help="Use cloud mode, default is False",
        action="store_true",
    )
    parser.add_argument(
        "-u",
        "--ui",
        type=str,
        default="cloud",
        help="User Inferface: 'mobile' for mobile phone (for training); 'local' for local hmi; 'cloud' for no UI",
    )
    parser.add_argument(
        "-r",
        "--resume",
        default=True,
        help="resume the last training with restored model, checkpoint and pedal map",
        action="store_true",
    )

    parser.add_argument(
        "-i",
        "--infer",
        default=False,
        help="No model update and training. Only Inference",
        action="store_true",
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
    parser.add_argument(
        "-v",
        "--vehicle",
        type=str,
        default=".",
        help="vehicle ID like 'VB7' or 'MP3' or VIN 'HMZABAAH1MF011055'",
    )
    parser.add_argument(
        "-d",
        "--driver",
        type=str,
        default=".",
        help="driver ID like 'longfei.zheng' or 'jiangbo.wei'",
    )
    parser.add_argument(
        "-m",
        "--remotecan",
        type=str,
        default="10.0.64.78:5000",
        help="url for remote can server, e.g. 10.10.0.6:30865, or name, e.g. baiduyun_k8s, newrizon_test",
    )
    parser.add_argument(
        "-w",
        "--web",
        type=str,
        default="10.0.64.78:9876",
        help="url for web ui server, e.g. 10.10.0.13:9876, or name, e.g. baiduyun_k8s, newrizon_test",
    )
    parser.add_argument(
        "-o",
        "--mongodb",
        type=str,
        default="mongo_local",
        help="url for mongodb server in format usr:password@host:port, e.g. admint:y02ydhVqDj3QFjT@10.10.0.4:23000, or simply name with synced default config, e.g. mongo_cluster, mongo_local",
    )
    args = parser.parse_args()

    # set up data folder (logging, checkpoint, table)

    try:
        app = RealtimeRDPG(
            args.cloud,
            args.ui,
            args.resume,
            args.infer,
            args.record_table,
            args.path,
            args.vehicle,
            args.driver,
            args.remotecan,
            args.web,
            args.mongodb,
            projroot,
            logger,
        )
    except TypeError as e:
        logger.error(f"Project Exeception TypeError: {e}", extra=dictLogger)
        sys.exit(1)
    except Exception as e:
        logger.error(e, extra=dictLogger)
        sys.exit(1)
    app.run()
