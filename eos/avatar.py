"""
Title: agent
Author: [Binjian Xin](https://www.newrizon.com)
Date created: 2022/12/14
Last modified: 2022/12/14
Description: Implement realtime reinforcement learning algorithm for training and inference
convergence of ddpg and rdpg agent
## Introduction

This script shows an implementation of rl agent on EC1 truck real environment.


An Ego Vehicle drives through a fixed track and collect loss (negative reward) defined
as energy consumption

### References

- [DDPG ](https://keras.io/examples/rl/ddpg_pendulum/)
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional, Union

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
import threading
import time
import warnings
import re
from functools import reduce

# third party imports
from collections import deque
from datetime import datetime
from logging.handlers import SocketHandler
from pathlib import Path, PurePosixPath
from threading import Lock, Thread

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# tf.debugging.set_log_device_placement(True)
# visualization import
import tensorflow as tf
from git import Repo
from pythonjsonlogger import jsonlogger  # type: ignore

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
from tensorflow.python.client import device_lib
from rocketmq.client import Message, Producer  # type: ignore

from eos import projroot
from eos.utils import dictLogger, logger
from eos.comm import RemoteCan, kvaser_send_float_array, ClearablePullConsumer
from eos.data_io.config import (
    trucks_by_id,
    trucks_by_vin,
    RE_VIN,
    can_servers_by_host,
    can_servers_by_name,
    trip_servers_by_name,
    trip_servers_by_host,
    drivers_by_id,
    Driver,
    RE_DRIVER,
    TruckInCloud,
    TruckInField,
)

from eos.comm import RemoteCan, ClearablePullConsumer, kvaser_send_float_array
from eos.utils import ragged_nparray_list_interp, GracefulKiller
from eos.visualization import plot_3d_figure, plot_to_image
from .algo import DPG

# from bson import ObjectId


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


# local imports


# from utils import get_logger, get_truck_status, flash_vcu, plot_3d_figure
# value = [99.0] * 21 * 17
# send_float_array('TQD_trqTrqSetECO_MAP_v', value)

# system warnings and numpy warnings handling
warnings.filterwarnings('ignore', message='currentThread', category=DeprecationWarning)
# np.warnings.filterwarnings('ignore', category=DeprecationWarning)


@dataclass
class Avatar(abc.ABC):
    ui: str = 'cloud'
    resume: bool = True
    infer_mode: bool = False
    record: bool = True
    path: str = '.'
    vehicle_str: str = 'HMZABAAH7MF011058'  # "VB7",
    driver_str: str = 'zheng-longfei'
    remotecan_srv: str = 'can_intra'
    web_srv: str = 'rocket_intra'
    pool_key: str = 'mongo_local'
    proj_root: Path = Path('.')
    logger: Optional[logging.Logger] = None
    data_root: Optional[Path] = None
    driver: Optional[Driver] = None
    truck: Optional[TruckInCloud | TruckInField] = None
    cloud: Optional[bool] = None  # determined by truck type
    _agent: Optional[DPG] = None  # set by derived Avartar like AvatarDDPG

    def __post_init__(
        self,
    ):
        self.repo = Repo(self.proj_root)
        # assert self.repo.is_dirty() == False, "Repo is dirty, please commit first"
        print(
            f"project root: {self.proj_root}, git head: {str(self.repo.head.commit)[:7]}, "
            f"author: {self.repo.head.commit.author}, "
            f"git message: {self.repo.head.commit.message}"
        )
        print(f"vehicle: {self.vehicle_str}")
        print(f"driver: {self.driver_str}")

        # Regex for VIN: HMZABAAH\wMF\d{6}
        p = re.compile(RE_VIN)
        if p.match(self.vehicle_str):
            # validate truck id
            # assert self.vehicle in self.trucks_by_vin.keys()
            self.truck = trucks_by_vin.get(self.vehicle_str)
            assert self.truck is not None, f'No Truck with VIN {self.vehicle_str}'
        else:
            # validate truck id
            # assert self.vehicle in self.trucks_by_id.keys()
            self.truck = trucks_by_id.get(self.vehicle_str)
            assert self.truck is not None, f'No Truck with name {self.vehicle_str}'
            if type(self.truck) == TruckInCloud:
                self.cloud = True
            else:
                self.cloud = False

        self.dictLogger = dictLogger
        # self.dictLogger = {"user": inspect.currentframe().f_code.co_name}

        # Regex for driver: ^[a-z]{1,10}[-,_][a-z]{1,10}(\d?){1,2}$
        p = re.compile(RE_DRIVER)
        # validate driver id
        assert p.match(
            self.driver_str
        ), "Driver name must be in format: String1[-,_,.]String2"
        self.driver = drivers_by_id.get(self.driver_str)
        assert self.driver is not None, f"Driver with ID {self.driver_str} not found!"
        if self.resume:
            self.data_root = projroot.joinpath(
                'data/' + self.truck.vin + '−' + self.driver.pid
            ).joinpath(self.path)
        else:
            self.data_root = projroot.joinpath(
                'data/scratch/' + self.truck.vin + '−' + self.driver.pid
            ).joinpath(self.path)

        self.set_logger()
        self.logc.info(f"{{\'header\': \'Start Logging\'}}", extra=self.dictLogger)
        self.logc.info(
            f"{{\'project_root\': \'{self.proj_root}\', "
            f"\'git_head\': {str(self.repo.head.commit)[:7]}, "
            f"\'author\': \'{self.repo.head.commit.author}\', "
            f"\'git_message\': \'{self.repo.head.commit.message}\'}}",
            extra=self.dictLogger,
        )
        self.logc.info(f"{{\'vehicle\': \'{self.truck.vid}\'}}", extra=self.dictLogger)
        self.logc.info(f"{{\'driver\': \'{self.driver.pid}\'}}", extra=self.dictLogger)

        self.eps = np.finfo(
            np.float32
        ).eps.item()  # smallest number such that 1.0 + eps != 1.0

        if self.cloud:
            # reset proxy (internal site force no proxy)
            self.init_cloud()
            assert self.ui in [
                'cloud',
                'local',
                'mobile',
            ], f'ui must be cloud, local or mobile, not {self.ui}'
            if self.ui == 'mobile':
                self.logger.info(
                    f"{{\'header\': \'Use phone UI\'}}", extra=self.dictLogger
                )
                self.get_truck_status = self.remote_webhmi_state_machine
            elif self.ui == 'local':
                self.logger.info(
                    f"{{\'header\': \'Use local UI\'}}", extra=self.dictLogger
                )
                self.get_truck_status = self.remote_hmi_state_machine
            elif self.ui == 'cloud':
                self.logger.info(
                    f"{{\'header\': \'Use cloud UI\'}}", extra=self.dictLogger
                )
                self.get_truck_status = self.remote_cloudhmi_state_machine
            else:
                raise ValueError("Unknown HMI type")
            self.flash_vcu = self.remote_flash_vcu
        else:
            self.get_truck_status = self.kvaser_get_truck_status
            self.flash_vcu = self.kvaser_flash_vcu

        self.logc.info(
            f"{{\'header\': \'Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}\'}}"
        )
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        self.logc.info(f'Tensorflow version: {tf.__version__}')
        tf_sys_details = tf.sysconfig.get_build_info()
        self.logc.info(f"{{\'header\': \'Tensorflow build info: {tf_sys_details}\'}}")

        self.set_data_path()
        tf.keras.backend.set_floatx('float32')
        self.logc.info(
            f"{{\'header\': \'tensorflow device lib:\n{device_lib.list_local_devices()}\'}}",
            extra=self.dictLogger,
        )
        self.logc.info(
            f"{{\'header\': \'Tensorflow Imported!\'}}", extra=self.dictLogger
        )

        self.init_vehicle()
        # DYNAMIC: need to adapt the pointer to change different roi of the pm, change the starting row index
        self.vcu_calib_table_row_start = 0
        self.logc.info(
            f"{{\'header\': \'VCU and GPU Initialization done!\'}}",
            extra=self.dictLogger,
        )
        self.init_threads_data()
        self.logc.info(
            f"{{\'header\': \'Thread data Initialization done!\'}}",
            extra=self.dictLogger,
        )

    @property
    def agent(self) -> Union[DPG | None]:
        return self._agent

    @agent.setter
    def agent(self, agent: DPG) -> None:
        self._agent = agent

    def init_cloud(self):
        os.environ['http_proxy'] = ''
        self.can_server = can_servers_by_name.get(self.remotecan_srv)
        if self.can_server is None:
            self.can_server = can_servers_by_host.get(self.remotecan_srv.split(':')[0])
            assert (
                self.can_server is not None
            ), f'No such remotecan host {self.remotecan_srv} found!'
            assert (
                self.remotecan_srv.split(':')[1] == self.can_server.Port
            ), f'Port mismatch for remotecan host {self.remotecan_srv}!'
        self.logc.info(f'CAN Server found: {self.remotecan_srv}', extra=self.dictLogger)

        self.remotecan_client = RemoteCan(
            truckname=self.truck.vid,
            url='http://' + self.can_server.Host + ':' + self.can_server.Port + '/',
        )

        if self.ui == 'mobile':
            self.trip_server = trip_servers_by_name.get(self.web_srv)
            if self.trip_server is None:
                self.trip_server = trip_servers_by_host.get(self.web_srv.split(':')[0])
                assert (
                    self.trip_server is not None
                ), f'No such trip server {self.web_srv} found!'
                assert (
                    self.web_srv.split(':')[1] == self.trip_server.Port
                ), f'Port mismatch for trip host {self.web_srv}!'
            self.logger.info(
                f'Trip Server found: {self.trip_server}', extra=self.dictLogger
            )

            # Create RocketMQ consumer
            self.rmq_consumer = ClearablePullConsumer('CID_EPI_ROCKET')
            self.rmq_consumer.set_namesrv_addr(
                self.trip_server.Host + ':' + self.trip_server.Port
            )

            # Create RocketMQ producer
            self.rmq_message_ready = Message('update_ready_state')
            self.rmq_message_ready.set_keys('what is keys mean')
            self.rmq_message_ready.set_tags('tags ------')
            self.rmq_message_ready.set_body(
                json.dumps({'vin': self.truck.vin, 'is_ready': True})
            )
            # self.rmq_message_ready.set_keys('trip_server')
            # self.rmq_message_ready.set_tags('tags')
            self.rmq_producer = Producer('PID-EPI_ROCKET')
            self.rmq_producer.set_namesrv_addr(
                self.trip_server.Host + ':' + self.trip_server.Port
            )

    def set_logger(self):
        self.logroot = self.data_root.joinpath('py_logs')
        try:
            os.makedirs(self.logroot)
        except FileExistsError:
            print('User folder exists, just resume!')

        logfilename = self.logroot.joinpath(
            'eos-rt-'
            + str(self.agent)
            + '-'
            + self.truck.vid
            + '-'
            + self.driver.pid
            + '-'
            + datetime.now().isoformat().replace(':', '-')
            + '.log'
        )
        fmt = '%(created)f-%(asctime)s.%(msecs)03d-%(name)s-'
        '%(levelname)s-%(module)s-%(threadName)s-%(funcName)s)-%(lineno)d): %(message)s'
        formatter = logging.Formatter(fmt)
        logging.basicConfig(
            format=fmt,
            datefmt='%Y-%m-%dT%H:%M:%S.%f',
        )
        logging.basicConfig(
            format=fmt,
            datefmt='%Y-%m-%dT%H:%M:%S.%f',
        )
        json_file_formatter = jsonlogger.JsonFormatter(
            '%(created)f %(asctime)s %(name)s '
            '%(levelname)s %(module)s %(threadName)s %(funcName)s) %(lineno)d) %(message)s'
        )

        fh = logging.FileHandler(logfilename)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(json_file_formatter)
        # strfilename = PurePosixPath(logfilename).stem + ".json"
        strfilename = self.logroot.joinpath(PurePosixPath(logfilename).stem + '.json')
        strh = logging.FileHandler(strfilename, mode='a')
        strh.setLevel(logging.DEBUG)
        strh.setFormatter(json_file_formatter)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        #  Cutelog socket
        skh = SocketHandler('127.0.0.1', 19996)
        skh.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(strh)
        self.logger.addHandler(ch)
        self.logger.addHandler(skh)

        self.logger.setLevel(logging.DEBUG)
        # self.dictLogger = {'funcName': '__self__.__func__.__name__'}
        # self.dictLogger = {'user': inspect.currentframe().f_back.f_code.co_name}

        self.logc = logger.getChild('main')  # main thread control flow
        self.logc.propagate = True
        # self.logd = logger.getChild("data flow")
        # self.logd.propagate = True
        self.tflog = tf.get_logger()
        self.tflog.addHandler(fh)
        self.tflog.addHandler(ch)
        self.tflog.addHandler(skh)
        self.tflog.addHandler(strh)

        self.tableroot = self.data_root.joinpath('tables')
        try:
            os.makedirs(self.tableroot)
        except FileExistsError:
            print('Table folder exists, just resume!')

    def set_data_path(self):
        # Create folder for ckpts loggings.
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.train_log_dir = self.data_root.joinpath(
            'tf_logs-'
            + str(self.agent)
            + self.truck.vid
            + '/gradient_tape/'
            + current_time
            + '/train'
        )
        self.train_summary_writer = tf.summary.create_file_writer(
            str(self.train_log_dir)
        )
        # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        if self.resume:
            self.logger.info(
                f"{{\'header\': \'Resume last training\'}}", extra=self.dictLogger
            )
        else:
            self.logger.info(
                f"{{\'header\': \'Start from scratch\'}}", extra=self.dictLogger
            )

    def init_vehicle(self):
        if self.resume:
            files = self.data_root.glob('last_table*.csv')
            if not files:
                self.logger.info(
                    f"{{\'header\': \'No last table found, start from default calibration table\'}}",
                    extra=self.dictLogger,
                )
                latest_file = (
                    self.proj_root / 'eos/data_io/config' / 'vb7_init_table.csv'
                )
            else:
                self.logger.info(
                    f"{{\'header\': \'Resume last table\'}}", extra=self.dictLogger
                )
                latest_file = max(files, key=os.path.getctime)

            self.vcu_calib_table0 = pd.read_csv(latest_file, index_col=0)

        else:
            self.logger.info(
                f"{{\'header\': \'Use default calibration table\'}}",
                extra=self.dictLogger,
            )
            latest_file = self.proj_root / 'eos/data_io/config' / 'vb7_init_table.csv'

        self.vcu_calib_table0 = pd.read_csv(latest_file, index_col=0)

        # pandas deep copy of the default table (while numpy shallow copy is sufficient)
        self.vcu_calib_table1 = self.vcu_calib_table0.copy(deep=True)
        self.logger.info(
            f"{{\'header\': \'Start flash initial table\'}}", extra=self.dictLogger
        )
        # time.sleep(1.0)
        if self.cloud:
            ret_code, ret_str = self.remotecan_client.send_torque_map(
                pedalmap=self.vcu_calib_table1, swap=False
            )  # 14 rows for whole map
            self.logger.info(
                f"{{\'header\': \'Done flash initial table.\',"
                f"\'ret_code\': {ret_code}\', "
                f"\'ret_str\': {ret_str}\'}}",
                extra=self.dictLogger,
            )
        else:
            ret_code = kvaser_send_float_array(self.vcu_calib_table1, sw_diff=False)
            self.logger.info(
                f"{{\'header\': \'Done flash initial table\', "
                f"\'ret_code\': {ret_code}\'}}",
                extra=self.dictLogger,
            )

        # TQD_trqTrqSetECO_MAP_v

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
                self.truck.cloud_unit_number
                * self.truck.cloud_unit_duration  # extend capture time after valid episode temrination
            )
        else:
            self.epi_countdown_time = (
                self.truck.kvaser_countdown  # extend capture time after valid episode temrination (3s)
            )

        # use timer object
        # self.timer_capture_countdown = threading.Timer(
        #     self.capture_countdown, self.capture_countdown_handler
        # )
        # signal.signal(signal.SIGALRM, self.reset_capture_handler)
        self.get_truck_status_start = False
        self.epi_countdown = False
        self.get_truck_status_motpow_t = []
        self.get_truck_status_myHost = '127.0.0.1'
        self.get_truck_status_myPort = 8002
        self.get_truck_status_qobject_len = 12  # sequence length 1.5*12s

    def capture_countdown_handler(
        self,
        evt_epi_done: threading.Event,
        evt_remote_get: threading.Event,
        evt_remote_flash: threading.Event,
    ):
        logger_countdown = self.logger.getChild('countdown')
        logger_countdown.propagate = True
        th_exit = False
        while not th_exit:
            with self.hmi_lock:
                if self.program_exit:
                    th_exit = True
                    continue

            logger_countdown.info(
                f"{{\'header\': \'wait for countdown\'}}", extra=self.dictLogger
            )
            evt_epi_done.wait()
            with self.done_env_lock:
                evt_epi_done.clear()
            # if episode is done, sleep for the extension time
            time.sleep(self.epi_countdown_time)
            # cancel wait as soon as waking up
            logger_countdown.info(
                f"{{\'header\': \'finish countdown\'}}", extra=self.dictLogger
            )

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
                f"{{\'header\': \'Episode done! free remote_flash and remote_get!\'}}",
                extra=self.dictLogger,
            )
            if self.cloud is False:
                self.vel_hist_dQ.clear()
            # raise Exception("reset capture to stop")
        logger_countdown.info(
            f"{{\'header\': \'Coutndown dies!!!\'}}", extra=self.dictLogger
        )

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
        logger_kvaser_get = self.logger.getChild('kvaser_get')
        logger_kvaser_get.propagate = True

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        socket.socket.settimeout(s, None)
        s.bind((self.get_truck_status_myHost, self.get_truck_status_myPort))
        # s.listen(5)
        logger_kvaser_get.info(
            f"{{\'header\': \'Socket Initialization Done!\'}}", extra=self.dictLogger
        )

        self.vel_hist_dQ = deque(maxlen=20)  # accumulate 1s of velocity values
        # vel_cycle_dQ = deque(maxlen=30)  # accumulate 1.5s (one cycle) of velocity values
        vel_cycle_dQ = deque(
            maxlen=self.truck.observation_length
        )  # accumulate 1.5s (one cycle) of velocity values
        with self.hmi_lock:
            self.program_start = True

        while not th_exit:  # th_exit is local; program_exit is global
            with self.hmi_lock:  # wait for tester to kick off or to exit
                if self.program_exit == True:  # if program_exit is True, exit thread
                    logger_kvaser_get.info(
                        f"{{\'header\': \'Capture thread exit due to processing request!!!\'}}",
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
                    f"{{\'header\': \'udp sending wrong data type!\'}}",
                    extra=self.dictLogger,
                )
                raise TypeError('udp sending wrong data type!')

            for key, value in pop_data.items():
                if key == 'status':  # state machine chores
                    # print(candata)
                    if value == 'begin':
                        self.get_truck_status_start = True
                        logger_kvaser_get.info(
                            f"{{\'header\': \'Episode will start!!!\'}}",
                            extra=self.dictLogger,
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

                    elif value == 'end_valid':
                        # DONE for valid end wait for another 2 queue objects (3 seconds) to get the last reward!
                        # cannot sleep the thread since data capturing in the same thread, use signal alarm instead
                        self.get_truck_status_start = (
                            True  # do not stopping data capture immediately
                        )

                        # set flag for countdown thread
                        with self.done_env_lock:
                            evt_epi_done.set()
                        logger_kvaser_get.info(
                            f"{{\'header\': \'Episode end starts countdown!\'}}"
                        )
                        with self.hmi_lock:
                            # self.episode_count += 1  # valid round increments self.epi_countdown = False
                            self.episode_done = False  # TODO delay episode_done to make main thread keep running
                            self.episode_end = False
                    elif value == 'end_invalid':
                        self.get_truck_status_start = False
                        logger_kvaser_get.info(
                            f"{{\'header\': \'Episode is interrupted!!!\'}}",
                            extra=self.dictLogger,
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
                    elif value == 'exit':
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
                elif key == 'data':
                    # self.logger.info('Data received before Capture starting!!!', extra=self.dictLogger)
                    # self.logger.info(f'ts:{value["timestamp"]}vel:{value["velocity"]}ped:{value["pedal"]}', extra=self.dictLogger)
                    # DONE add logic for episode valid and invalid
                    try:
                        if self.get_truck_status_start:  # starts episode
                            ts = datetime.now().timestamp()
                            velocity = float(value['velocity'])
                            pedal = float(value['pedal'])
                            brake = float(value['brake_pressure'])
                            current = float(value['A'])
                            voltage = float(value['V'])

                            motion_power = [
                                ts,
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
                                >= self.truck.observation_length
                            ):
                                if len(vel_cycle_dQ) != vel_cycle_dQ.maxlen:
                                    self.logc.warning(  # the recent 1.5s average velocity
                                        f"{{\'header\': \'cycle deque is inconsistent!\'}}",
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
                                        f"{{\'header\': \'cycle higher than 120km/h!\'}}",
                                        extra=self.dictLogger,
                                    )
                                    self.vcu_calib_table_row_start = 16
                                # get the row of the table

                                logger_kvaser_get.info(
                                    f"{{\'header\': \'Cycle velocity\', "
                                    f"\'aver\': {vel_aver:.2f}, "
                                    f"\'min\': {vel_min:.2f}, "
                                    f"\'max\': {vel_max:.2f}, "
                                    f"\'start_index\': {self.vcu_calib_table_row_start}\'}}",
                                    extra=self.dictLogger,
                                )
                                # self.logc.info(
                                #     f"Producer Queue has {motionpowerQueue.qsize()}!", extra=self.dictLogger,
                                # )
                                df_motion_power = pd.DataFrame(
                                    self.get_truck_status_motpow_t,
                                    columns=[
                                        'timestep',
                                        'velocity',
                                        'thrust',
                                        'brake',
                                        'current',
                                        'voltage',
                                    ],
                                )
                                # df_motion_power.set_index('timestamp', inplace=True)
                                df_motion_power.columns.name = 'qtuple'

                                with self.captureQ_lock:
                                    self.motionpowerQueue.put(df_motion_power)
                                    motionpowerQueue_size = (
                                        self.motionpowerQueue.qsize()
                                    )
                                logger_kvaser_get.info(
                                    f"{{\'header\': \'motionpowerQueue size: {motionpowerQueue_size}\'}}",
                                    extra=self.dictLogger,
                                )
                                self.get_truck_status_motpow_t = []
                    except Exception as e:
                        logger_kvaser_get.info(
                            f"{{\'header\': \'kvaser get signal error\',"
                            f"\'exception\': \'{e}\'}}",  # f"Valid episode, Reset data capturing to stop after 3 seconds!",
                            extra=self.dictLogger,
                        )
                        break
                else:
                    logger_kvaser_get.warning(
                        f"{{\'header\': \'udp sending message with key: {key}; value: {value}\'}}"
                    )

                    break

        logger_kvaser_get.info(
            f"{{\'header\': \'get_truck_status dies!!!\'}}", extra=self.dictLogger
        )

        s.close()

    # this is the calibration table consumer for flashing
    # @eye
    def kvaser_flash_vcu(self, evt_remote_flash: threading.Event):
        flash_count = 0
        th_exit = False

        logger_flash = self.logger.getChild('kvaser_flash')
        logger_flash.propagate = True

        logger_flash.info(
            f"{{\'header\': \'Initialization Done!\'}}", extra=self.dictLogger
        )
        while not th_exit:
            # time.sleep(0.1)
            with self.hmi_lock:
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
                        self.truck.torque_row_num_flash,
                        self.truck.torque_table_col_num,
                    ],
                )

                # get change budget : % of initial table
                vcu_calib_table_reduced = (
                    vcu_calib_table_reduced * self.truck.torque_budget
                )

                # dynamically change table row start index
                vcu_calib_table0_reduced = self.vcu_calib_table0.to_numpy()[
                    table_start : self.truck.torque_row_num_flash + table_start,
                    :,
                ]
                vcu_calib_table_min_reduced = (
                    vcu_calib_table0_reduced - self.truck.torque_budget
                )  # apply the budget instead of truck.torque_lower_bound
                vcu_calib_table_max_reduced = (
                    self.truck.torque_upper_bound * vcu_calib_table0_reduced
                )  # 1.0*

                vcu_calib_table_reduced = tf.clip_by_value(
                    vcu_calib_table_reduced + vcu_calib_table0_reduced,
                    clip_value_min=vcu_calib_table_min_reduced,
                    clip_value_max=vcu_calib_table_max_reduced,
                )

                # create updated complete pedal map, only update the first few rows
                # vcu_calib_table1 keeps changing as the cache of the changing pedal map
                self.vcu_calib_table1.iloc[
                    table_start : self.truck.torque_table_row_num_flash + table_start
                ] = vcu_calib_table_reduced.numpy()

                if args.record_table:
                    curr_table_store_path = self.tableroot.joinpath(
                        'instant_table_'
                        + str(self.agent)
                        + '-'
                        + self.truck.vid
                        + '-'
                        + self.driver.pid
                        + '-'
                        + datetime.now().strftime('%y-%m-%d-%h-%m-%s-')
                        + 'e-'
                        + str(epi_cnt)
                        + '-'
                        + str(step_count)
                        + '.csv'
                    )
                    with open(curr_table_store_path, 'wb') as f:
                        self.vcu_calib_table1.to_csv(curr_table_store_path)
                        # np.save(last_table_store_path, vcu_calib_table1)
                    logger_flash.info(
                        f"{{\'header\': \'E{epi_cnt} done with record instant table: {step_count}\'}}",
                        extra=self.dictLogger,
                    )

                logger_flash.info(
                    f"{{\'header\': \'flash starts\'}}", extra=self.dictLogger
                )
                ret_code = kvaser_send_float_array(self.vcu_calib_table1, sw_diff=True)
                # time.sleep(1.0)

                if ret_code != 0:
                    logger_flash.error(
                        f"{{\'header\': \'kvaser_send_float_array failed: {ret_code}\'}}",
                        extra=self.dictLogger,
                    )
                else:
                    logger_flash.info(
                        f"{{\'header\': \'flash done, count:{flash_count}\'}}",
                        extra=self.dictLogger,
                    )
                    flash_count += 1
                # watch(flash_count)

        logger_flash.info(
            f"{{\'header\': \'Save the last table!!!!\'}}", extra=self.dictLogger
        )
        last_table_store_path = (
            self.data_root.joinpath(  # there's no slash in the end of the string
                'last_table_'
                + str(self.agent)
                + '-'
                + self.truck.vid
                + '-'
                + self.driver.pid
                + '-'
                + datetime.now().strftime('%y-%m-%d-%H-%M-%S')
                + '.csv'
            )
        )
        with open(last_table_store_path, 'wb') as f:
            self.vcu_calib_table1.to_csv(last_table_store_path)
        logger_flash.info(
            f"{{\'header\': \'flash_vcu dies!!!\'}}", extra=self.dictLogger
        )

    def remote_get_handler(
        self,
        evt_remote_get: threading.Event,
        evt_remote_flash: threading.Event,
    ):
        th_exit = False
        logger_remote_get = self.logger.getChild('remote_get')
        logger_remote_get.propagate = True

        while not th_exit:
            with self.hmi_lock:
                if self.program_exit:
                    th_exit = self.program_exit
                    continue
                episode_end = self.episode_end
            if episode_end is True:
                logger_remote_get.info(
                    f"{{\'header\': \'Episode ends and wait for evt_remote_get!\'}}",
                    extra=self.dictLogger,
                )
                with self.get_env_lock:
                    evt_remote_get.clear()
                # continue

            logger_remote_get.info(
                f"{{\'header\': \'wait for remote get trigger\'}}",
                extra=self.dictLogger,
            )
            evt_remote_get.wait()

            # after long wait, need to refresh state machine
            with self.hmi_lock:
                th_exit = self.program_exit
                episode_end = self.episode_end

            if episode_end is True:
                logger_remote_get.info(
                    f"{{\'header\': \'Episode ends after evt_remote_get without get_signals!\'}}",
                    extra=self.dictLogger,
                )
                with self.get_env_lock:
                    evt_remote_get.clear()
                continue

            # if episode is done, sleep for the extension time
            # cancel wait as soon as waking up
            timeout = self.truck.cloud_unit_number + 7
            logger_remote_get.info(
                f"{{\'header\': \'Wake up to fetch remote data\', "
                f"\'duration\': {self.truck.cloud_unit_number}, "
                f"\'timeout\': {timeout}}}",
                extra=self.dictLogger,
            )
            with self.remoteClient_lock:
                (ret_code, ret_msg) = self.remotecan_client.get_signals(
                    duration=self.truck.cloud_unit_number, timeout=timeout
                )  # timeout is 1 second longer than duration
                if ret_code != 0:  # in case of failure, ping server
                    logger_remote_get.warning(
                        f"{{\'header\': \'RemoteCAN failure!\',"
                        f"\'ret_code\': {ret_code}, "
                        f"\'ret_msg\'={ret_msg}\'}}",
                        extra=self.dictLogger,
                    )

                    response = os.system('ping -c 1 ' + self.can_server.Host)
                    if response == 0:
                        logger_remote_get.info(
                            f"{{\'header\': \'host is up\', "
                            f"\'host\': \'{self.can_server.Host}\'}}",
                            extra=self.dictLogger,
                        )
                    else:
                        logger_remote_get.info(
                            f"{{\'header\': \'host is down\', "
                            f"\'host\': \'{self.can_server.Host}\'}}",
                            extra=self.dictLogger,
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

            if not isinstance(ret_msg, dict):
                logger_remote_get.critical(
                    f"{{\'header\': \'udp sending wrong data type!\'}}",
                    extra=self.dictLogger,
                )
                raise TypeError('udp sending wrong data type!')
            else:
                logger_remote_get.info(
                    f"{{\'header\': \'Get remote data, signal_success={ret_code}!\'}}",
                    extra=self.dictLogger,
                )

            try:
                if ret_code == 0:
                    with self.hmi_lock:
                        th_exit = self.program_exit
                        episode_end = self.episode_end
                    if episode_end is True:
                        logger_remote_get.info(
                            f"{{\'header\': \'Episode ends, not waiting for evt_remote_flash and continue!\'}}",
                            extra=self.dictLogger,
                        )
                        with self.get_env_lock:
                            evt_remote_get.clear()
                        continue

                    try:
                        signal_freq = self.truck.cloud_signal_frequency
                        gear_freq = self.truck.cloud_gear_frequency
                        unit_duration = self.truck.cloud_unit_duration
                        unit_ob_num = unit_duration * signal_freq
                        unit_gear_num = unit_duration * gear_freq
                        unit_num = self.truck.cloud_unit_number
                        for key, value in ret_msg.items():
                            if key == 'result':
                                logger_remote_get.info(
                                    "{{\'header\': \'convert observation state to array.\'}}",
                                    extra=self.dictLogger,
                                )
                                # timestamp processing
                                timestamps_list = []
                                separators = '--T::.'  # adaption separators of the raw intest string
                                start_century = '20'
                                for ts in value['timestamps']:
                                    # create standard iso string datetime format
                                    ts_substrings = [
                                        ts[i : i + 2] for i in range(0, len(ts), 2)
                                    ]
                                    ts_iso = start_century
                                    for i, sep in enumerate(separators):
                                        ts_iso = ts_iso + ts_substrings[i] + sep
                                    ts_iso = ts_iso + ts_substrings[-1]
                                    timestamps_list.append(
                                        ts_iso
                                    )  # string of timestamps in iso format, UTC-0
                                timestamps_units = list(
                                    (
                                        np.array(timestamps_list).astype(
                                            'datetime64[ms]'
                                        )  # convert to milliseconds
                                        - np.timedelta64(
                                            8, 'h'
                                        )  # to np.datetime64 (in local time UTC-8)
                                    ).astype(  # convert to UTC+8  TODO using pytz.timezone for conversion
                                        'int'
                                    )  # convert to int, time unit is millisecond
                                )  # convert to list of int
                                if len(timestamps_units) != unit_num:
                                    raise ValueError(
                                        f'timestamps_units length is {len(timestamps_units)}, not {unit_num}'
                                    )
                                # upsample gears from 2Hz to 50Hz
                                sampling_interval = 1.0 / signal_freq * 1000  # in ms
                                timestamps_list = [
                                    i + j * sampling_interval
                                    for i in timestamps_units
                                    for j in np.arange(unit_ob_num)
                                ]
                                timestamps = np.array(timestamps_list).reshape(
                                    (self.truck.cloud_unit_number, -1)
                                )  # final format is a list of integers as timestamps in ms
                                current = ragged_nparray_list_interp(
                                    value['list_current_1s'],
                                    ob_num=unit_ob_num,  # 4s * 1s * 50Hz
                                )  # 4x50
                                voltage = ragged_nparray_list_interp(
                                    value['list_voltage_1s'],
                                    ob_num=unit_ob_num,
                                )
                                thrust = ragged_nparray_list_interp(
                                    value['list_pedal_1s'],
                                    ob_num=unit_ob_num,
                                )
                                brake = ragged_nparray_list_interp(
                                    value['list_brake_pressure_1s'],
                                    ob_num=unit_ob_num,
                                )
                                velocity = ragged_nparray_list_interp(
                                    value['list_speed_1s'],
                                    ob_num=unit_ob_num,
                                )  # 4*50
                                gear = ragged_nparray_list_interp(
                                    value['list_gears'],
                                    ob_num=unit_gear_num,
                                )
                                # upsample gears from 2Hz to 50Hz
                                gear = np.repeat(
                                    gear,
                                    (signal_freq // gear_freq),
                                    axis=1,
                                )

                                idx = pd.DatetimeIndex(
                                    timestamps.flatten(), tz=self.truck.tz
                                )
                                df_motion_power = pd.DataFrame(
                                    {
                                        'timestep': timestamps.flatten(),
                                        'velocity': velocity.flatten(),
                                        'thrust': thrust.flatten(),
                                        'brake': brake.flatten(),
                                        'gear': gear.flatten(),
                                        'current': current.flatten(),
                                        'voltage': voltage.flatten(),
                                    },
                                )
                                df_motion_power.columns.name = 'qtuple'
                                # df_motion_power.set_index('timestamp', inplace=True)
                                # motion_power = np.c_[
                                #     timestamps.reshape(-1, 1),  # 200
                                #     velocity.reshape(-1, 1),  # 200
                                #     thrust.reshape(-1, 1),  # 200
                                #     brake.reshape(-1, 1),  # 200
                                #     gears.reshape(-1, 1),  # 200
                                #     current.reshape(-1, 1),  # 200
                                #     voltage.reshape(-1, 1),  # 200
                                # ]  # 1 + 3 + 1 + 2  : im 7  # 200*7

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
                                        f"{{\'header\': \'cycle higher than 120km/h!\'}}",
                                        extra=self.dictLogger,
                                    )
                                    self.vcu_calib_table_row_start = 16

                                logger_remote_get.info(
                                    f"{{\'header\': \'Cycle velocity description\', "
                                    f"\'aver\': {np.mean(velocity):.2f}, "
                                    f"\'min\': {np.amin(velocity):.2f}, "
                                    f"\'max\': {np.amax(velocity):.2f}, "
                                    f"\'start_index\': {self.vcu_calib_table_row_start}\'}}",
                                    extra=self.dictLogger,
                                )

                                with self.captureQ_lock:
                                    self.motionpowerQueue.put(df_motion_power)

                                logger_remote_get.info(
                                    f"{{\'header\': \'Get one record, wait for remote_flash!!!\'}}",
                                    extra=self.dictLogger,
                                )
                                # as long as one observation is received, always waiting for flash
                                evt_remote_flash.wait()
                                with self.flash_env_lock:
                                    evt_remote_flash.clear()
                                logger_remote_get.info(
                                    f"{{\'header\': \'evt_remote_flash wakes up, "
                                    f"reset inner lock, restart remote_get!!!\'}}",
                                    extra=self.dictLogger,
                                )
                            else:
                                # self.logger.info(
                                #     f"show status: {key}:{value}",
                                #     extra=self.dictLogger,
                                # )
                                pass
                    except Exception as e:
                        logger_remote_get.error(
                            f"{{\'header\': \'Observation Corrupt! \', "
                            f"\'exception\': {e}\'}}",
                            extra=self.dictLogger,
                        )
                else:
                    logger_remote_get.error(
                        f"{{\'header\': \'get_signals failed: {ret_msg}\'}}",
                        extra=self.dictLogger,
                    )

            except Exception as e:
                logger_remote_get.info(
                    f"{{\'header\': \'Break due to Exception\', "
                    f"\'exception\': {e}\'}}",
                    extra=self.dictLogger,
                )

            with self.get_env_lock:
                evt_remote_get.clear()

        logger_remote_get.info(
            f"{{\'header\': \'thr_remoteget dies!!!!!\'}}", extra=self.dictLogger
        )

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
        logger_webhmi_sm = self.logger.getChild('webhmi_sm')
        logger_webhmi_sm.propagate = True
        th_exit = False

        try:
            self.rmq_consumer.start()
            self.rmq_producer.start()
            logger_webhmi_sm.info(
                f"{{\'header\': \'Start RocketMQ client\', "
                f"\'host\': \'{self.trip_server.Host}\'}}",
                extra=self.dictLogger,
            )

            msg_topic = self.driver.pid + '_' + self.truck.vid

            broker_msgs = self.rmq_consumer.pull(msg_topic)
            logger_webhmi_sm.info(
                f"{{\'header\': \'Before clearing history: Pull\', "
                f"\'msg_number\': {len(list(broker_msgs))}, "
                f"\'topic\': {msg_topic}\'}}",
                extra=self.dictLogger,
            )
            self.rmq_consumer.clear_history(msg_topic)
            broker_msgs = self.rmq_consumer.pull(msg_topic)
            logger_webhmi_sm.info(
                f"{{\'header\': \'After clearing history: Pull\', "
                f"\'msg_number\': {len(list(broker_msgs))}, "
                f"\'topic\': {msg_topic}\'}}",
                extra=self.dictLogger,
            )
            all(broker_msgs)  # exhaust history messages

        except Exception as e:
            logger_webhmi_sm.error(
                f"{{\'header\': \'send_sync failed\', " f"\'exception\': {e}\'}}",
                extra=self.dictLogger,
            )
            return
        try:
            # send ready signal to trip server
            ret = self.rmq_producer.send_sync(self.rmq_message_ready)
            logger_webhmi_sm.info(
                f"{{\'header\': \'Sending ready signal to trip server\', "
                f"\'status\': \'{ret.status}\', "
                f"\'msg_id\': \'{ret.msg_id}\', "
                f"\'offset\': \'{ret.offset}\'}}",
                extra=self.dictLogger,
            )
            with self.state_machine_lock:
                self.program_start = True

            logger_webhmi_sm.info(
                f"{{\'header\': \'RocketMQ client Initialization Done!\'}}",
                extra=self.dictLogger,
            )
        except Exception as e:
            logger_webhmi_sm.error(
                f"{{\'header\': \'Fatal Failure!\', " f"\'exception\': {e}\'}}",
                extra=self.dictLogger,
            )
            return

        while not th_exit:  # th_exit is local; program_exit is global
            with self.hmi_lock:  # wait for tester to kick off or to exit
                if self.program_exit == True:  # if program_exit is True, exit thread
                    logger_webhmi_sm.info(
                        f"{{\'header\': \'Capture thread exit due to processing request!!!\'}}",
                        extra=self.dictLogger,
                    )
                    th_exit = True
                    continue
            msgs = self.rmq_consumer.pull(msg_topic)
            for msg in msgs:
                msg_body = json.loads(msg.body)
                if not isinstance(msg_body, dict):
                    logger_webhmi_sm.critical(
                        f"{{\'header\': \'rocketmq server sending wrong data type!\'}}",
                        extra=self.dictLogger,
                    )
                    raise TypeError('rocketmq server sending wrong data type!')
                logger_webhmi_sm.info(
                    f"{{\'header\': \'Get message\', " f"\'msg\': {msg_body}\'}}",
                    extra=self.dictLogger,
                )
                if msg_body['vin'] != self.truck.vin:
                    continue

                if msg_body['code'] == 5:  # "config/start testing"
                    logger_webhmi_sm.info(
                        f"{{\'header\': \'Restart/Reconfigure message\', "
                        f"\'VIN\': \'{msg_body['vin']}\', "
                        f"\'driver\': \'{msg_body['name']}\'}}",
                        extra=self.dictLogger,
                    )

                    with self.state_machine_lock:
                        self.program_start = True

                    # send ready signal to trip server
                    ret = self.rmq_producer.send_sync(self.rmq_message_ready)
                    logger_webhmi_sm.info(
                        f"{{\'header\': \'Sending ready signal to trip server\', "
                        f"\'status\': \'{ret.status}\', "
                        f"\'msg_id\': \'{ret.msg_id}\', "
                        f"\'offset\': \'{ret.offset}\'}}",
                        extra=self.dictLogger,
                    )
                elif msg_body['code'] == 1:  # start episode
                    self.get_truck_status_start = True
                    logger_webhmi_sm.info(
                        f"{{\'header\': \'Episode will start!!!\'}}",
                        extra=self.dictLogger,
                    )
                    th_exit = False
                    # ts_epi_start = time.time()
                    with self.get_env_lock:
                        evt_remote_get.clear()
                    with self.flash_env_lock:
                        evt_remote_flash.clear()
                    logger_webhmi_sm.info(
                        f"{{\'header\': \'Episode start! clear remote_flash and remote_get!\'}}",
                        extra=self.dictLogger,
                    )

                    with self.captureQ_lock:
                        while not self.motionpowerQueue.empty():
                            self.motionpowerQueue.get()
                    with self.hmi_lock:
                        self.episode_done = False
                        self.episode_end = False
                elif msg_body['code'] == 2:  # valid stop
                    # DONE for valid end wait for another 2 queue objects (3 seconds) to get the last reward!
                    # cannot sleep the thread since data capturing in the same thread, use signal alarm instead

                    logger_webhmi_sm.info(
                        f"{{\'header\': \'End Valid!!!!!!\'}}", extra=self.dictLogger
                    )
                    self.get_truck_status_start = (
                        True  # do not stopping data capture immediately
                    )

                    # set flag for countdown thread
                    with self.done_env_lock:
                        evt_epi_done.set()

                    logger_webhmi_sm.info(
                        f"{{\'header\': \'Episode end starts countdown!\'}}"
                    )
                    with self.hmi_lock:
                        # self.episode_count += 1  # valid round increments self.epi_countdown = False
                        self.episode_done = False  # TODO delay episode_done to make main thread keep running
                        self.episode_end = False
                elif msg_body['code'] == 3:  # invalid stop
                    self.get_truck_status_start = False
                    logger_webhmi_sm.info(
                        f"{{\'header\': \'Episode is interrupted!!!\'}}",
                        extra=self.dictLogger,
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
                        f"{{\'header\': \'end_invalid! free remote_flash and remote_get!\'}}",
                        extra=self.dictLogger,
                    )

                    with self.hmi_lock:
                        self.episode_done = False
                        self.episode_end = True
                        self.episode_count += 1  # invalid round increments
                elif msg_body['code'] == 4:  # "exit"
                    self.get_truck_status_start = False
                    self.get_truck_status_motpow_t = []

                    with self.get_env_lock:
                        evt_remote_get.set()
                    with self.flash_env_lock:
                        evt_remote_flash.set()
                    logger_webhmi_sm.info(
                        f"{{\'header\': \'Program exit!!!! free remote_flash and remote_get!\'}}",
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
                        f"{{\'header\': \'Unknown message\',"
                        f"\'msg_body\': {msg_body}\'}}",
                        extra=self.dictLogger,
                    )

            time.sleep(0.05)  # sleep for 50ms to update state machine
            if self.get_truck_status_start:
                with self.get_env_lock:
                    evt_remote_get.set()

        self.rmq_consumer.shutdown()
        self.rmq_producer.shutdown()
        logger_webhmi_sm.info(
            f"{{\'header\': \'remote webhmi dies!!!\'}}", extra=self.dictLogger
        )

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

        logger_cloudhmi_sm = self.logger.getChild('cloudhmi_sm')
        logger_cloudhmi_sm.propagate = True

        logger_cloudhmi_sm.info(
            f"{{\'header\': \'Start/Configure message\', "
            f"\'VIN\': {self.truck.vin}, "
            f"\'driver\': {self.driver.pid}\'}}",
            extra=self.dictLogger,
        )

        with self.state_machine_lock:
            self.program_start = True

        logger_cloudhmi_sm.info(
            f"{{\'header\': \'Road Test with inferring will start as one single episode!!!\'}}",
            extra=self.dictLogger,
        )
        with self.get_env_lock:
            evt_remote_get.clear()
        with self.flash_env_lock:
            evt_remote_flash.clear()

        with self.hmi_lock:
            self.episode_done = False
            self.episode_end = False

        while not th_exit:  # th_exit is local; program_exit is global
            with self.hmi_lock:  # wait for tester to kick off or to exit
                # Check if the runner is trying to kill the process
                # kill signal captured from main thread
                if self.program_exit == True:  # if program_exit is True, exit thread
                    logger_cloudhmi_sm.info(
                        f"{{\'header\': \'UI thread exit due to processing request!!!\'}}",
                        extra=self.dictLogger,
                    )

                    self.get_truck_status_start = False
                    self.get_truck_status_motpow_t = []

                    with self.get_env_lock:
                        evt_remote_get.set()
                    with self.flash_env_lock:
                        evt_remote_flash.set()
                    logger_cloudhmi_sm.info(
                        f"{{\'header\': \'Process is being killed and Program exit!!!! "
                        f"Free remote_flash and remote_get!\'}}",
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

            time.sleep(0.05)  # sleep for 50ms to update state machine
            with self.get_env_lock:
                evt_remote_get.set()

        logger_cloudhmi_sm.info(
            f"{{\'header\': \'remote cloudhmi killed gracefully!!!\'}}",
            extra=self.dictLogger,
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

        logger_hmi_sm = self.logger.getChild('hmi_sm')
        logger_hmi_sm.propagate = True
        #  Get the HMI control command from UDP, but not the data from KvaserCAN
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        socket.socket.settimeout(s, None)
        s.bind((self.get_truck_status_myHost, self.get_truck_status_myPort))
        # s.listen(5)
        logger_hmi_sm.info(
            f"{{\'header\': \'Socket Initialization Done!\'}}", extra=self.dictLogger
        )

        while not th_exit:  # th_exit is local; program_exit is global
            with self.hmi_lock:  # wait for tester to kick off or to exit
                if self.program_exit == True:  # if program_exit is True, exit thread
                    logger_hmi_sm.info(
                        f"{{\'header\': \'Capture thread exit due to processing request!!!\'}}",
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
                    f"{{\'header\': \'udp sending wrong data type!\'}}",
                    extra=self.dictLogger,
                )
                raise TypeError('udp sending wrong data type!')

            for key, value in pop_data.items():
                if key == 'status':  # state machine chores
                    # print(candata)
                    # self.logc.info(
                    #     f"Status data: key={key},value={value}!!!!!!", extra=self.dictLogger
                    # )
                    if value == 'begin':
                        self.get_truck_status_start = True
                        logger_hmi_sm.info(
                            f"{{\'header\': \'Episode will start!!!\'}}",
                            extra=self.dictLogger,
                        )
                        th_exit = False
                        # ts_epi_start = time.time()
                        with self.get_env_lock:
                            evt_remote_get.clear()
                        with self.flash_env_lock:
                            evt_remote_flash.clear()
                        logger_hmi_sm.info(
                            f"{{\'header\': \'Episode start! clear remote_flash and remote_get!\'}}",
                            extra=self.dictLogger,
                        )

                        with self.captureQ_lock:
                            while not self.motionpowerQueue.empty():
                                self.motionpowerQueue.get()
                        with self.hmi_lock:
                            self.episode_done = False
                            self.episode_end = False
                    elif value == 'end_valid':
                        # DONE for valid end wait for another 2 queue objects (3 seconds) to get the last reward!
                        # cannot sleep the thread since data capturing in the same thread, use signal alarm instead

                        logger_hmi_sm.info(
                            f"{{\'header\': \'End Valid!!!!!!\'}}",
                            extra=self.dictLogger,
                        )
                        self.get_truck_status_start = (
                            True  # do not stopping data capture immediately
                        )

                        # set flag for countdown thread
                        with self.done_env_lock:
                            evt_epi_done.set()
                        logger_hmi_sm.info(
                            f"{{\'header\': \'Episode end starts countdown!\'}}"
                        )
                        with self.hmi_lock:
                            # self.episode_count += 1  # valid round increments self.epi_countdown = False
                            self.episode_done = False  # TODO delay episode_done to make main thread keep running
                            self.episode_end = False
                    elif value == 'end_invalid':
                        self.get_truck_status_start = False
                        logger_hmi_sm.info(
                            f"{{\'header\': \'Episode is interrupted!!!\'}}",
                            extra=self.dictLogger,
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
                            f"{{\'header\': \'end_invalid! free remote_flash and remote_get!\'}}",
                            extra=self.dictLogger,
                        )

                        with self.hmi_lock:
                            self.episode_done = False
                            self.episode_end = True
                            self.episode_count += 1  # invalid round increments
                    elif value == 'exit':
                        self.get_truck_status_start = False
                        self.get_truck_status_motpow_t = []

                        with self.get_env_lock:
                            evt_remote_get.set()
                        with self.flash_env_lock:
                            evt_remote_flash.set()
                        logger_hmi_sm.info(
                            f"{{\'header\': \'Program exit!!!! free remote_flash and remote_get!\'}}",
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
                elif key == 'data':
                    #  instead of get kvasercan, we get remotecan data here!
                    if self.get_truck_status_start:  # starts episode
                        # set flag for remote_get thread
                        with self.get_env_lock:
                            evt_remote_get.set()
                        # self.logc.info(f"Kick off remoteget!!")
                else:
                    logger_hmi_sm.warning(
                        f"{{\'header\': \'udp sending message with\', "
                        f"\'key\': \'{key}\', "
                        f"\'value\': \'{value}\'}}"
                    )

                    break

        s.close()
        logger_hmi_sm.info(
            f"{{\'header\': \'remote hmi dies!!!\'}}", extra=self.dictLogger
        )

    def remote_flash_vcu(self, evt_remote_flash: threading.Event):
        """
        trigger 1: tableQueue is not empty
        trigger 2: remote client is available as signaled by the remote_get thread
        """
        flash_count = 0
        th_exit = False

        logger_flash = self.logger.getChild('flash')
        logger_flash.propagate = True
        logger_flash.info(
            f"{{\'header\': \'Initialization Done!\'}}", extra=self.dictLogger
        )
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
                        f"{{\'header\': \'Episode ends, skipping remote_flash and continue!\'}}",
                        extra=self.dictLogger,
                    )
                    continue
            except queue.Empty:
                pass
            else:
                vcu_calib_table_reduced = tf.reshape(
                    table,
                    [
                        self.truck.torque_row_num_flash,
                        self.truck.torque_table_col_num,
                    ],
                )

                # get change budget : % of initial table
                vcu_calib_table_reduced = (
                    vcu_calib_table_reduced * self.truck.torque_budget
                )

                # dynamically change table row start index
                vcu_calib_table0_reduced = self.vcu_calib_table0.to_numpy()[
                    table_start : self.truck.torque_row_num_flash + table_start,
                    :,
                ]
                vcu_calib_table_min_reduced = (
                    vcu_calib_table0_reduced - self.truck.torque_budget
                )
                vcu_calib_table_max_reduced = (
                    self.truck.torque_upper_bound * vcu_calib_table0_reduced
                )

                vcu_calib_table_reduced = tf.clip_by_value(
                    vcu_calib_table_reduced + vcu_calib_table0_reduced,
                    clip_value_min=vcu_calib_table_min_reduced,
                    clip_value_max=vcu_calib_table_max_reduced,
                )

                # create updated complete pedal map, only update the first few rows
                # vcu_calib_table1 keeps changing as the cache of the changing pedal map
                self.vcu_calib_table1.iloc[
                    table_start : self.truck.torque_row_num_flash + table_start
                ] = vcu_calib_table_reduced.numpy()

                if args.record_table:
                    curr_table_store_path = self.tableroot.joinpath(
                        'instant_table_'
                        + str(self.agent)
                        + '-'
                        + self.truck.vid
                        + '-'
                        + self.driver.pid
                        + '-'
                        + datetime.now().strftime('%y-%m-%d-%h-%m-%s-')
                        + 'e-'
                        + str(epi_cnt)
                        + '-'
                        + str(step_count)
                        + '.csv'
                    )
                    with open(curr_table_store_path, 'wb') as f:
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
                timeout = self.truck.torque_row_num_flash + 3
                logger_flash.info(
                    f"{{\'header\': \'flash starts\', " f"\'timeout\': {timeout}\'}}",
                    extra=self.dictLogger,
                )
                # lock doesn't control the logic explictitly
                # competetion is not desired
                with self.remoteClient_lock:
                    (ret_code, ret_str) = self.remotecan_client.send_torque_map(
                        pedalmap=self.vcu_calib_table1.iloc[
                            table_start : self.truck.torque_row_num_flash + table_start
                        ],
                        swap=False,
                        timeout=timeout,
                    )
                # time.sleep(1.0)
                if ret_code != 0:
                    logger_flash.error(
                        f"{{\'header\': \'send_torque_map failed and retry\', "
                        f"\'ret_code\': \'{ret_code}\', "
                        f"\'ret_str\': \'{ret_str}\'}}",
                        extra=self.dictLogger,
                    )

                    response = os.system('ping -c 1 ' + self.can_server.Url)
                    if response == 0:
                        logger_flash.info(
                            f"{{\'header\': \'Can server is up!\', "
                            f"\'host\': \'{self.can_server.Url}\'}}",
                            extra=self.dictLogger,
                        )
                    else:
                        logger_flash.info(
                            f"{{\'header\': \'Can server is down!\', "
                            f"\'host\': \'{self.can_server.Url}\'}}",
                            extra=self.dictLogger,
                        )
                else:
                    logger_flash.info(
                        f"{{\'header\': \'flash done\', "
                        f"\'count\': {flash_count}\'}}",
                        extra=self.dictLogger,
                    )
                    flash_count += 1

                # flash is done and unlock remote_get
                with self.flash_env_lock:
                    evt_remote_flash.set()

                # watch(flash_count)

        logger_flash.info(
            f"{{\'header\': \'Save the last table!!!!\'}}", extra=self.dictLogger
        )

        last_table_store_path = (
            self.data_root.joinpath(  # there's no slash in the end of the string
                'last_table_'
                + str(self.agent)
                + '-'
                + self.truck.vid
                + '-'
                + self.driver.pid
                + '-'
                + datetime.now().strftime('%y-%m-%d-%H-%M-%S')
                + '.csv'
            )
        )
        with open(last_table_store_path, 'wb') as f:
            self.vcu_calib_table1.to_csv(last_table_store_path)
        # motionpowerQueue.join()
        logger_flash.info(
            f"{{\'header\': \'remote_flash_vcu dies!!!\'}}", extra=self.dictLogger
        )

    def assemble_state_ser(self, motionpower: pd.DataFrame) -> pd.Series:
        """
        assemble state df from motionpower df
        order is vital for the model:
        "timestep, velocity, thrust, brake"
        contiguous storage in each measurement
        due to sort_index, output:
        [col0: brake, col1: thrust, col2: timestep, col3: velocity]
        """
        state = (
            motionpower.loc[:, ['timestep', 'velocity', 'thrust', 'brake']]
            .stack()
            .swaplevel(0, 1)
        )
        state.name = 'state'
        state.index.names = ['rows', 'idx']
        state.sort_index(
            inplace=True
        )  # sort by rows and idx (brake, thrust, timestep, velocity)

        return state

    def assemble_reward_ser(self, motionpower: pd.DataFrame) -> pd.Series:
        """
        assemble reward df from motionpower df
        order is vital for the model:
        contiguous storage in each row, due to sort_index, output:
        [timestep, work]
        """

        pow_t = motionpower.loc[:, ['current', 'voltage']]
        ui_sum = pow_t.prod(axis=1).sum()
        wh = (
            ui_sum / 3600.0 / self.truck.observation_sampling_rate
        )  # rate 0.05 for kvaser, 0.02 remote # negative wh
        self.logc.info(
            f'wh: {wh}',
            extra=self.dictLogger,
        )
        work = wh * (-1.0)
        reward_ts = pd.to_datetime(datetime.now())
        reward = (
            pd.DataFrame({'work': work, 'timestep': reward_ts}, index=[0])
            .stack()
            .swaplevel(0, 1)
            .sort_index()  # columns oder (timestep, work)
        )
        reward.name = 'reward'
        reward.index.names = ['rows', 'idx']
        return reward

    def assemble_action_ser(
        self,
        torque_map_line: tf.Tensor,
        table_start: int,
        flash_start_ts: pd.Timestamp,
        flash_end_ts: pd.Timestamp,
    ) -> pd.Series:
        """
        generate action df from torque_map_line
        order is vital for the model:
        contiguous storage in each row, due to sort_index, output:
        "r0, r1, r2, r3, ..., ,speed, throttle(map),timestep"
        """
        # assemble_action_df
        row_num = self.truck.action_flashrow
        speed_ser = pd.Series(
            self.truck.speed_scale[
                table_start : table_start + self.truck.action_flashrow
            ],
            name='speed',
        )
        throttle_ser = pd.Series(self.truck.pedal_scale, name='throttle')
        torque_map = tf.reshape(
            torque_map_line,
            [self.truck.torque_row_num_flash, self.truck.torque_table_col_num],
        )
        df_torque_map = pd.DataFrame(
            torque_map.to_numpy()
        ).transpose()  # row to columns
        df_torque_map.columns = (
            self.agent.torque_table_row_names
        )  # index: [r0, r1, ...]

        span_each_row = (flash_end_ts - flash_start_ts) / row_num
        flash_timestamps_ser = pd.Series(
            flash_start_ts
            + pd.date_range(
                np.linspace(1, row_num, row_num) * span_each_row, unit='ms'
            ),
            name='timestep',
        )
        action = (
            reduce(
                lambda left, right: pd.merge(
                    left,
                    right,
                    how='outer',
                    left_index=True,
                    right_index=True,
                ),
                [df_torque_map, flash_timestamps_ser, speed_ser, throttle_ser],
            )
            .stack()
            .swaplevel(0, 1)
            .sort_index()  # columns order (r0, r1, ..., speed, throttle, timestep)
        )
        action.name = 'action'
        action.index.names = ['rows', 'idx']
        action.columns.names = []

        return action

    def run(self):
        # Start thread for flashing vcu, flash first
        evt_epi_done = threading.Event()
        evt_remote_get = threading.Event()
        evt_remote_flash = threading.Event()
        self.thr_countdown = Thread(
            target=self.capture_countdown_handler,
            name='countdown',
            args=[evt_epi_done, evt_remote_get, evt_remote_flash],
        )
        self.thr_countdown.start()

        self.thr_observe = Thread(
            target=self.get_truck_status,
            name='observe',
            args=[evt_epi_done, evt_remote_get, evt_remote_flash],
        )
        self.thr_observe.start()

        if self.cloud:
            self.thr_remoteget = Thread(
                target=self.remote_get_handler,
                name='remoteget',
                args=[evt_remote_get, evt_remote_flash],
            )
            self.thr_remoteget.start()

        self.thr_flash = Thread(
            target=self.flash_vcu, name='flash', args=[evt_remote_flash]
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

        self.logc.info(f'main Initialization done!', extra=self.dictLogger)
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

            self.logc.info('----------------------', extra=self.dictLogger)
            self.logc.info(
                f"{{\'header\': \'episosde starts!\', " f"\'episode\': {epi_cnt}}}",
                extra=self.dictLogger,
            )

            # mongodb default to UTC time
            self.agent.start_episode(datetime.now())

            tf.debugging.set_log_device_placement(True)
            with tf.device('/GPU:0'):
                while (
                    not epi_end
                ):  # end signal, either the round ends normally or user interrupt
                    if killer.kill_now:
                        self.logc.info(f'Process is being killed!!!')
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
                    self.logc.info(f'motionpowerQueue.qsize(): {motionpowerqueue_size}')
                    if epi_end and done and (motionpowerqueue_size > 2):
                        # self.logc.info(f"motionpowerQueue.qsize(): {self.motionpowerQueue.qsize()}")
                        self.logc.info(
                            f"{{\'header\': \'Residue in Queue is a sign of disordered sequence, interrupted!\'}}"
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
                            f"{{\'header\': \'No data in the Queue!!!\', "
                            f"\'episode\': {epi_cnt}}}",
                            extra=self.dictLogger,
                        )
                        continue

                    self.logc.info(
                        f"{{\'header\': \'start\', "
                        f"\'step\': {step_count}, "
                        f"\'episode\': {epi_cnt}}}",
                        extra=self.dictLogger,
                    )  # env.step(action) action is flash the vcu calibration table

                    # !!!no parallel even!!!
                    # predict action probabilities and estimated future rewards
                    # from environment state
                    # for causal rl, the odd indexed observation/reward are caused by last action
                    # skip the odd indexed observation/reward for policy to make it causal

                    # assemble state
                    timestamp = motionpower.loc[
                        0, 'timestep'
                    ]  # only take the first timestamp, as frequency is fixed at 50Hz, the rest is saved in another col
                    state = self.assemble_state_ser(motionpower)

                    # assemble reward
                    reward = self.assemble_reward_ser(work)
                    work = reward[('work', 0)]
                    episode_reward += work

                    self.logc.info(
                        f"{{\'header\': \'assembling state and reward!\', "
                        f"\'episode\': {epi_cnt}}}",
                        extra=self.dictLogger,
                    )
                    # stripping timestamps from state, (later flatten and convert to tensor)
                    torque_map_line = self.agent.actor_predict(
                        state[['velocity', 'thrust', 'brake']], int(step_count / 1)
                    )  # model input requires fixed order velocity col -> thrust col -> brake col
                    #  !!! training with samples of the same order!!!

                    self.logc.info(
                        f"{{\'header\': \'inference done with reduced action space!\', "
                        f"\'episode\': {epi_cnt}}}",
                        extra=self.dictLogger,
                    )
                    # flash the vcu calibration table and assemble action
                    flash_start_ts = pd.to_datetime(datetime.now())
                    with self.tableQ_lock:
                        self.tableQueue.put(torque_map_line)
                        self.logc.info(
                            f"{{\'header\': \'Action Push table\', "
                            f"\'StartIndex\': {table_start}, "
                            f"\'qsize\': {self.tableQueue.qsize()}}}",
                            extra=self.dictLogger,
                        )

                    # wait for remote flash to finish
                    evt_remote_flash.wait()
                    flash_end_ts = pd.to_datetime(datetime.now())

                    action = self.assemble_action_ser(
                        torque_map_line,
                        table_start,
                        flash_start_ts,
                        flash_end_ts,
                    )

                    if step_count > 0:
                        self.agent.deposit(
                            prev_timestamp,
                            prev_state,
                            prev_action,
                            reward,
                            state,
                        )

                    prev_timestamp = timestamp
                    prev_state = state
                    prev_action = action

                    # TODO add speed sum as positive reward
                    self.logc.info(
                        f"{{\'header\': \'Step done\',"
                        f"\'step\': {step_count}, "
                        f"\'episode\': {epi_cnt}}}",
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
                    f"{{\'header\': \'interrupted, waits for next episode to kick off!\' "
                    f"\'episode\': {epi_cnt}}}",
                    extra=self.dictLogger,
                )
                # send ready signal to trip server
                if self.ui == 'mobile':
                    ret = self.rmq_producer.send_sync(self.rmq_message_ready)
                    self.logc.info(
                        f"{{\'header\': \'Sending ready signal to trip server\', "
                        f"\'status\': \'{ret.status}\', "
                        f"\'msg-id\': \'{ret.msg_id}\', "
                        f"\'offset\': \'{ret.offset}\'}}",
                        extra=self.dictLogger,
                    )
                continue  # otherwise assuming the history is valid and back propagate

            self.agent.end_episode()  # deposit history

            self.logger.info(
                f"{{\'header\': \'Episode end.\', " f"\'episode\': {epi_cnt}, ",
                f"\'timestamp\': {datetime.now()}}}",
                extra=self.dictLogger,
            )

            critic_loss = 0
            actor_loss = 0
            if self.infer_mode:
                (critic_loss, actor_loss) = self.agent.get_losses()
                # FIXME bugs in maximal sequence length for ungraceful testing
                # self.logc.info("Nothing to be done for rdgp!")
                self.logc.info(
                    "{{\'header\': \'No Learning, just calculating loss.\'}}"
                )
            else:
                self.logc.info("{{\'header\': \'Learning and updating 6 times!\'}}")
                for k in range(6):
                    # self.logger.info(f"BP{k} starts.", extra=self.dictLogger)
                    if self.agent.buffer.count() > 0:
                        (critic_loss, actor_loss) = self.agent.train()
                        self.agent.soft_update_target()
                    else:
                        self.logc.info(
                            f"{{\'header\': \'Buffer empty, no learning!\'}}",
                            extra=self.dictLogger,
                        )
                        self.logc.info(
                            '++++++++++++++++++++++++', extra=self.dictLogger
                        )
                        break
                # Checkpoint manager save model
                self.agent.save_ckpt()

            self.logc.info(
                f"{{\'header\': \'losses after 6 times BP\', "
                f"\'episode\': {epi_cnt}, "
                f"\'critic loss\': {critic_loss}, "
                f"\'actor loss\': {actor_loss}}}",
                extra=self.dictLogger,
            )

            # update running reward to check condition for solving
            running_reward = 0.05 * (-episode_reward) + (1 - 0.05) * running_reward

            # Create a matplotlib 3d figure, //export and save in log
            fig = plot_3d_figure(self.vcu_calib_table1)

            # tf logging after episode ends
            # use local episode counter epi_cnt_local tf.summary.writer; otherwise specify multiple self.logdir and automatic switch
            with self.train_summary_writer.as_default():
                tf.summary.scalar('WH', -episode_reward, step=epi_cnt_local)
                tf.summary.scalar('actor loss', actor_loss, step=epi_cnt_local)
                tf.summary.scalar('critic loss', critic_loss, step=epi_cnt_local)
                tf.summary.scalar('reward', episode_reward, step=epi_cnt_local)
                tf.summary.scalar('running reward', running_reward, step=epi_cnt_local)
                tf.summary.image(
                    'Calibration Table', plot_to_image(fig), step=epi_cnt_local
                )
                tf.summary.histogram(
                    'Calibration Table Hist',
                    self.vcu_calib_table1.to_numpy().tolist(),
                    step=epi_cnt_local,
                )
                # tf.summary.trace_export(
                #     name="veos_trace", step=epi_cnt_local, profiler_outdir=train_log_dir
                # )

            epi_cnt_local += 1
            plt.close(fig)

            self.logc.info(
                f"{{\'episode\': {epi_cnt}, " f"\'reward\': {episode_reward}}}",
                extra=self.dictLogger,
            )

            self.logc.info("----------------------", extra=self.dictLogger)
            if epi_cnt % 10 == 0:
                self.logc.info("++++++++++++++++++++++++", extra=self.dictLogger)
                self.logc.info(
                    f"{{\'header\': \'Running reward\': {running_reward:.2f}, "
                    f"\'episode\': \'{epi_cnt}\'}}",
                    extra=self.dictLogger,
                )
                self.logc.info("++++++++++++++++++++++++", extra=self.dictLogger)

            # send ready signal to trip server
            if self.ui == 'mobile':
                ret = self.rmq_producer.send_sync(self.rmq_message_ready)
                self.logger.info(
                    f"{{\'header\': \'Sending ready signal to trip server\', "
                    f"\'status\': \'{ret.status}\', "
                    f"\'msg_id\': \'{ret.msg_id}\', "
                    f"\'offset\': \'{ret.offset}\'}}",
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

        self.logc.info(f"{{\'header\': \'main dies!!!!\'}}", extra=self.dictLogger)


if __name__ == '__main__':
    """
    ## Setup
    """
    # resumption settings
    parser = argparse.ArgumentParser(
        'Use RL agent (DDPG or RDPG) with tensorflow backend for EOS with coastdown activated and expected velocity in 3 seconds'
    )
    parser.add_argument(
        '-a',
        '--agent',
        type=str,
        default='ddpg',
        help="RL agent choice: 'ddpg' for DDPG; 'rdpg' for Recurrent DPG",
    )

    parser.add_argument(
        '-c',
        '--cloud',
        default=False,
        help='Use cloud mode, default is False',
        action='store_true',
    )

    parser.add_argument(
        '-u',
        '--ui',
        type=str,
        default='cloud',
        help="User Inferface: 'mobile' for mobile phone (for training); 'local' for local hmi; 'cloud' for no UI",
    )

    parser.add_argument(
        '-r',
        '--resume',
        default=True,
        help='resume the last training with restored model, checkpoint and pedal map',
        action='store_true',
    )

    parser.add_argument(
        '-i',
        '--infer',
        default=False,
        help='No model update and training. Only Inference mode',
        action='store_true',
    )
    parser.add_argument(
        '-t',
        '--record_table',
        default=True,
        help='record action table during training',
        action='store_true',
    )
    parser.add_argument(
        '-p',
        '--path',
        type=str,
        default='.',
        help='relative path to be saved, for create subfolder for different drivers',
    )
    parser.add_argument(
        '-v',
        '--vehicle',
        type=str,
        default='.',
        help="vehicle ID like 'VB7' or 'MP3' or VIN 'HMZABAAH1MF011055'",
    )
    parser.add_argument(
        '-d',
        '--driver',
        type=str,
        default='.',
        help="driver ID like 'longfei.zheng' or 'jiangbo.wei'",
    )
    parser.add_argument(
        '-m',
        '--remotecan',
        type=str,
        default='10.0.64.78:5000',
        help='url for remote can server, e.g. 10.10.0.6:30865, or name, e.g. baiduyun_k8s, newrizon_test',
    )
    parser.add_argument(
        '-w',
        '--web',
        type=str,
        default='10.0.64.78:9876',
        help='url for web ui server, e.g. 10.10.0.13:9876, or name, e.g. baiduyun_k8s, newrizon_test',
    )
    parser.add_argument(
        '-o',
        '--mongodb',
        type=str,
        default='mongo_local',
        help="url for mongodb server in format usr:password@host:port, e.g. admint:y02ydhVqDj3QFjT@10.10.0.4:23000, or simply name with synced default config, e.g. mongo_cluster, mongo_local; if specified as empty string '', use local npy file",
    )
    args = parser.parse_args()

    # set up data folder (logging, checkpoint, table)

    assert args.agent in ['ddpg', 'rdpg'], 'agent must be either ddpg or rdpg'
    try:
        app = Avatar(
            args.agent,
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
        logger.error(
            f"{{\'header\': \'Project Exeception TypeError\', "
            f"\'exception\': \'{e}\'}}",
            extra=dictLogger,
        )
        sys.exit(1)
    except Exception as e:
        logger.error(
            f"{{\'header\': \main Exeception\', " f"\'exception\': \'{e}\'}}",
            extra=dictLogger,
        )
        sys.exit(1)
    app.run()
