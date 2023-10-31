"""
Title: avatar
Author: [Binjian Xin](https://www.newrizon.com)
Date created: 2022/12/14
Last modified: 2022/12/14
Description: Implement realtime reinforcement learning algorithm for training and inference
convergence of ddpg and rdpg agent
## Introduction

This script shows an implementation of rl agent on EC1 truck real environment.


An Ego Vehicle drives through a fixed track and collect loss (negative reward) defined
as energy consumption

"""
from __future__ import annotations
import concurrent.futures

import abc
import argparse
import json

# logging
import logging
import math

# system imports
import os
import sys
from threading import Thread, Event
import time
import warnings
import concurrent.futures

# third party imports
from dataclasses import dataclass, field
from datetime import datetime
from logging.handlers import SocketHandler
from pathlib import Path, PurePosixPath
from typing import Optional, Union, cast
from typeguard import check_type  # type: ignore

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd  # type: ignore

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

# tf.debugging.set_log_device_placement(True)
# visualization import
import tensorflow as tf
from git import Repo
from pythonjsonlogger import jsonlogger  # type: ignore
from tensorflow.summary import SummaryWriter, create_file_writer, scalar  # type: ignore

from eos import proj_root
from eos.agent import DDPG, DPG, RDPG
from eos.agent.utils import HyperParamDDPG, HyperParamRDPG
from eos.data_io.config import (
    Driver,
    Truck,
    TruckInCloud,
    str_to_can_server,
    str_to_driver,
    str_to_trip_server,
    str_to_truck,
)

from eos.utils import (
    GracefulKiller,
    dictLogger,
    logger,
)

from eos.data_io.dataflow import Pipeline, Producer, Consumer, Kvaser, Cruncher


from eos.visualization import plot_3d_figure, plot_to_image


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# system warnings and numpy warnings handling
warnings.filterwarnings("ignore", message="currentThread", category=DeprecationWarning)
# np.warnings.filterwarnings('ignore', category=DeprecationWarning)


@dataclass(kw_only=True)
class Avatar(abc.ABC):
    _truck: Truck
    _driver: Driver
    _agent: DPG  # set by derived Avatar like AvatarDDPG
    logger: logging.Logger
    dictLogger: dict
    _observe_pipeline: Optional[Pipeline] = None
    _crunch_pipeline: Optional[Pipeline] = None
    _flash_pipeline: Optional[Pipeline] = None
    start_event: Optional[Event] = None
    stop_event: Optional[Event] = None
    interrupt_event: Optional[Event] = None
    countdown_event: Optional[Event] = None
    exit_event: Optional[Event] = None
    kvaser: Optional[Kvaser] = None
    cruncher: Optional[Cruncher] = None
    data_root: Path = Path(".") / "data"
    table_root: Path = Path(".") / "tables"
    log_root: Path = Path(".") / "py_log"
    logger_control_flow: Optional[logging.Logger] = None
    tflog: Optional[logging.Logger] = None

    def __post_init__(
        self,
    ) -> None:
        self.repo = Repo(proj_root)
        # assert self.repo.is_dirty() == False, "Repo is dirty, please commit first"
        short_sha = self.repo.git.rev_parse(self.repo.head.commit.hexsha, short=7)
        print(
            f"Project root: {proj_root}, "  # type: ignore
            f"git head: {short_sha}, "
            f"author: {self.repo.head.commit.author.name}, "
            f"git message: {self.repo.head.commit.message}"
        )
        self.dictLogger = dictLogger
        # self.dictLogger = {"user": inspect.currentframe().f_code.co_name}

        self.set_logger()  # define self.logger and self.logger_control_flow
        self.logger_control_flow.info(
            f"{{'header': 'Start Logging'}}", extra=self.dictLogger
        )
        self.logger_control_flow.info(
            f"{{'project_root': '{proj_root}', "  # type: ignore
            f"'git_head': {short_sha}, "
            f"'author': '{self.repo.head.commit.author.name}', "
            f"'git_message': '{self.repo.head.commit.message}'}}",
            extra=self.dictLogger,
        )
        self.logger_control_flow.info(
            f"{{'vehicle': '{self.truck.vid}'}}", extra=self.dictLogger
        )
        self.logger_control_flow.info(
            f"{{'driver': '{self.driver.pid}'}}", extra=self.dictLogger
        )

        self.eps = np.finfo(
            np.float32
        ).eps.item()  # smallest number such that 1.0 + eps != 1.0

        # if self.cloud:
        #     self.pipeline = CloudPipeline()
        #     # reset proxy (internal site force no proxy)
        # else:
        #     self.pipeline = EcuPipeline()

        # misc variables required for the class and its methods
        # self.vel_hist_dq: deque = deque(maxlen=20)  # type: ignore
        # self.program_start = False
        # self.program_exit = False
        # self.hmi_lock = threading.Lock()

        self.logger_control_flow.info(
            f"{{'header': 'Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}'}}"
        )
        # gpus = tf.config.list_physical_devices(device_type="GPU")
        # tf.config.experimental.set_memory_growth(gpus[0], True)
        self.logger_control_flow.info(f"Tensorflow version: {tf.__version__}")
        tf_sys_details = tf.sysconfig.get_build_info()
        self.logger_control_flow.info(
            f"{{'header': 'Tensorflow build info: {tf_sys_details}'}}"
        )

        self.set_data_path()
        tf.keras.backend.set_floatx("float32")
        self.logger_control_flow.info(
            f"{{'header': 'tensorflow device lib:\n{tf.config.list_physical_devices()}'}}",
            extra=self.dictLogger,
        )
        self.logger_control_flow.info(
            f"{{'header': 'Tensorflow Imported!'}}", extra=self.dictLogger
        )

        self.init_vehicle()
        self.logger_control_flow.info(
            f"{{'header': 'VCU Initialization done!'}}",
            extra=self.dictLogger,
        )
        # DYNAMIC: need to adapt the pointer to change different roi of the pm, change the starting row index
        self.vcu_calib_table_row_start = 0
        self.logger_control_flow.info(
            f"{{'header': 'Thread data Initialization done!'}}",
            extra=self.dictLogger,
        )

        # initialize dataflow: pipelines, producers, consumers, crunchers
        self.observe_pipeline = Pipeline(buffer_size=3)  # pipeline for observations
        self.flash_pipeline = Pipeline(
            buffer_size=3
        )  # pipeline for flashing torque tables
        self.start_event = Event()
        self.stop_event = Event()
        self.interrupt_event = Event()
        self.countdown_event = Event()
        self.exit_event = Event()
        self.kvaser = Kvaser(
            truck=self.truck,
            driver=self.driver,
            in_pipeline=self.observe_pipeline,
            in_start_event=self.start_event,
            in_stop_event=self.stop_event,
            in_interrupt_event=self.interrupt_event,
            in_countdown_evt=self.countdown_evt,
            in_exit_event=self.exit_event,
            out_pipeline=self.flash_pipeline,
            out_start_event=self.start_event,
            out_stop_event=self.stop_event,
            out_interrupt_event=self.interrupt_event,
            out_countdown_evt=self.countdown_evt,
            out_exit_event=self.exit_event,
            logger=self.logger,
            dictLogger=self.dictLogger,
        )
        self.cruncher = Cruncher(
            truck=self.truck,
            driver=self.driver,
            out_pipeline=self.observe_pipeline,
            out_start_event=self.start_event,
            out_stop_event=self.stop_event,
            out_interrupt_event=self.interrupt_event,
            out_exit_event=self.exit_event,
            logger=self.logger,
            dictLogger=self.dictLogger,
        )

    @property
    def agent(self) -> Union[DPG, None]:
        return self._agent

    @agent.setter
    def agent(self, value: DPG) -> None:
        self._agent = value

    @property
    def truck(self) -> Union[Truck, None]:
        return self._truck

    @truck.setter
    def truck(self, value: Truck) -> None:
        self._truck = value

    @property
    def driver(self) -> Union[Driver, None]:
        return self._driver

    @driver.setter
    def driver(self, value: Driver) -> None:
        self._driver = value

    @property
    def observe_pipeline(self) -> Union[Pipeline, None]:
        return self._observe_pipeline

    @observe_pipeline.setter
    def observe_pipeline(self, value: Pipeline) -> None:
        self._observe_pipeline = value

    @property
    def crunch_pipeline(self) -> Union[Pipeline, None]:
        return self._crunch_pipeline

    @crunch_pipeline.setter
    def crunch_pipeline(self, value: Pipeline) -> None:
        self._crunch_pipeline = value

    @property
    def flash_pipeline(self) -> Union[Pipeline, None]:
        return self._flash_pipeline

    @flash_pipeline.setter
    def flash_pipeline(self, value: Pipeline) -> None:
        self._flash_pipeline = value

    def set_logger(self):
        self.log_root = self.data_root / "py_logs"
        try:
            os.makedirs(self.log_root)
        except FileExistsError:
            print("User folder exists, just resume!")

        log_file_name = self.log_root.joinpath(
            "eos-rt-"
            + str(self.agent)
            + "-"
            + self.truck.vid
            + "-"
            + self.driver.pid
            + "-"
            + pd.Timestamp.now(self.truck.tz).isoformat()  # .replace(":", "-")
            + ".log"
        )
        fmt = "%(asctime)s-%(name)s-%(levelname)s-%(module)s-%(threadName)s-%(funcName)s)-%(lineno)d): %(message)s"
        formatter = logging.Formatter(fmt)
        logging.basicConfig(
            format=fmt,
            datefmt="%Y-%m-%dT%H:%M:%S.%f",
        )
        json_file_formatter = jsonlogger.JsonFormatter(
            "%(created)f %(asctime)s %(name)s "
            "%(levelname)s %(module)s %(threadName)s %(funcName)s) %(lineno)d) %(message)s"
        )

        file_handler = logging.FileHandler(log_file_name)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(json_file_formatter)
        # str_file_name = PurePosixPath(log_file_name).stem + ".json"
        str_file_name = self.log_root.joinpath(
            PurePosixPath(log_file_name).stem + ".json"
        )
        str_handler = logging.FileHandler(str_file_name, mode="a")
        str_handler.setLevel(logging.DEBUG)
        str_handler.setFormatter(json_file_formatter)

        char_handler = logging.StreamHandler()
        char_handler.setLevel(logging.DEBUG)
        char_handler.setFormatter(formatter)
        #  Cutelog socket
        socket_handler = SocketHandler("127.0.0.1", 19996)
        socket_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(str_handler)
        self.logger.addHandler(char_handler)
        self.logger.addHandler(socket_handler)

        self.logger.setLevel(logging.DEBUG)

        self.logger_control_flow = logger.getChild("main")  # main thread control flow
        self.logger_control_flow.propagate = True
        self.tflog = tf.get_logger()
        self.tflog.addHandler(file_handler)
        self.tflog.addHandler(char_handler)
        self.tflog.addHandler(socket_handler)
        self.tflog.addHandler(str_handler)

        self.table_root = self.data_root.joinpath("tables")
        try:
            os.makedirs(self.table_root)
        except FileExistsError:
            print("Table folder exists, just resume!")

    def set_data_path(self) -> None:
        # Create folder for ckpts logs.
        current_time = pd.Timestamp.now(self.truck.tz).isoformat()
        train_log_dir = self.data_root.joinpath(
            "tf_logs-"
            + str(self.agent)
            + self.truck.vid
            + "/gradient_tape/"
            + current_time
            + "/train"
        )
        self.train_summary_writer: SummaryWriter = create_file_writer(  # type: ignore
            str(train_log_dir)
        )
        # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        if self.resume:
            self.logger.info(
                f"{{'header': 'Resume last training'}}", extra=self.dictLogger
            )
        else:
            self.logger.info(
                f"{{'header': 'Start from scratch'}}", extra=self.dictLogger
            )

    def init_vehicle(self) -> None:
        if self.resume:
            files = sorted(self.data_root.glob("last_table*.csv"))
            if not files:
                self.logger.info(
                    f"{{'header': 'No last table found, start from default calibration table'}}",
                    extra=self.dictLogger,
                )
                latest_file = proj_root / "eos/data_io/config" / "vb7_init_table.csv"
            else:
                self.logger.info(
                    f"{{'header': 'Resume last table'}}", extra=self.dictLogger
                )
                latest_file = max(files, key=os.path.getctime)

        else:
            self.logger.info(
                f"{{'header': 'Use default calibration table'}}",
                extra=self.dictLogger,
            )
            latest_file = proj_root / "eos/data_io/config" / "vb7_init_table.csv"

        self.vcu_calib_table0 = pd.read_csv(latest_file, index_col=0)

        # pandas deep copy of the default table (while numpy shallow copy is sufficient)
        self.vcu_calib_table1 = self.vcu_calib_table0.copy(deep=True)
        self.logger.info(
            f"{{'header': 'Start flash initial table'}}", extra=self.dictLogger
        )
        # time.sleep(1.0)
        if self.cloud:
            ret_code, ret_str = self.pipeline.send_torque_map(
                pedalmap=self.vcu_calib_table1, swap=False
            )  # 14 rows for whole map
            self.logger.info(
                f"{{'header': 'Done flash initial table.',"
                f"'ret_code': {ret_code}', "
                f"'ret_str': {ret_str}'}}",
                extra=self.dictLogger,
            )
        else:
            ret_code = self.pipeline.kvaser_send_float_array(
                self.vcu_calib_table1, sw_diff=False
            )
            self.logger.info(
                f"{{'header': 'Done flash initial table', "
                f"'ret_code': {ret_code}'}}",
                extra=self.dictLogger,
            )


if __name__ == "__main__":
    """
    ## Setup
    """
    # resumption settings
    parser = argparse.ArgumentParser(
        "Use RL agent (DDPG or RDPG) with tensorflow backend for EOS with coast-down activated "
        "and expected velocity in 3 seconds"
    )
    parser.add_argument(
        "-a",
        "--agent",
        type=str,
        default="ddpg",
        help="RL agent choice: 'ddpg' for DDPG; 'rdpg' for Recurrent DPG",
    )

    parser.add_argument(
        "-c",
        "--cloud",
        default=False,
        help="Use cloud mode, default is False",
        action="store_true",
    )

    parser.add_argument(
        "-u",
        "--ui",
        type=str,
        default="TCP",
        help="User Interface: "
        "'RMQ' for mobile phone (using rocketmq for training/assessment); "
        "'TCP' for local hmi (using loopback tcp for training/assessment); "
        "'NUMB' for non-interaction for inference only and testing purpose",
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
        "--infer_mode",
        default=False,
        help="No model update and training. Only Inference mode",
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
        "--data_path",
        type=str,
        default=".",
        help="relative path to be saved, for create sub-folder for different drivers",
    )
    parser.add_argument(
        "-v",
        "--vehicle",
        type=str,
        default="VB7_FIELD",
        help="vehicle ID like 'VB7' or 'MP3' or VIN 'HMZABAAH1MF011055'",
    )
    parser.add_argument(
        "-d",
        "--driver",
        type=str,
        default="wang-kai",
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
        "--pool_key",
        type=str,
        default="mongo_local",
        help="pool selection for data storage, "
        "url for mongodb server in format usr:password@host:port, e.g. admint:y02ydhVqDj3QFjT@10.10.0.4:23000, "
        "or simply name with synced default config, e.g. mongo_cluster, mongo_local; "
        "if specified as path name, use dask local under proj_root/data folder or cluster",
    )
    args = parser.parse_args()

    # set up data folder (logging, checkpoint, table)
    try:
        truck: Truck = str_to_truck(args.vehicle)
    except KeyError:
        raise KeyError(f"vehicle {args.vehicle} not found in config file")
    else:
        logger.info(
            f"Vehicle found. vid:{truck.vid}, vin: {truck.vin}.", extra=dictLogger
        )

    try:
        driver: Driver = str_to_driver(args.driver)
    except KeyError:
        raise KeyError(f"driver {args.driver} not found in config file")
    else:
        logger.info(
            f"Driver found. pid:{driver.pid}, vin: {driver.name}.", extra=dictLogger
        )

    # remotecan_srv: str = 'can_intra'
    try:
        can_server = str_to_can_server(args.remotecan)
    except KeyError:
        raise KeyError(f"can server {args.remotecan} not found in config file")
    else:
        logger.info(f"CAN Server found: {can_server.SRVName}", extra=dictLogger)

    try:
        trip_server = str_to_trip_server(args.web)
    except KeyError:
        raise KeyError(f"trip server {args.web} not found in config file")
    else:
        logger.info(f"Trip Server found: {trip_server.SRVName}", extra=dictLogger)

    assert args.agent in ["ddpg", "rdpg"], "agent must be either ddpg or rdpg"

    if args.resume:
        data_root = proj_root.joinpath("data/" + truck.vin + "-" + driver.pid).joinpath(
            args.data_path
        )
    else:  # from scratch
        data_root = proj_root.joinpath(
            "data/scratch" + truck.vin + "-" + driver.pid
        ).joinpath(args.data_path)

    if args.agent == "ddpg":
        agent: DDPG = DDPG(
            _coll_type="RECORD",
            _hyper_param=HyperParamDDPG(),
            _truck=truck,
            _driver=driver,
            _pool_key=args.pool_key,
            _data_folder=str(data_root),
            _infer_mode=args.infer_mode,
            _resume=args.resume,
        )
    else:  # args.agent == 'rdpg':
        agent: RDPG = RDPG(  # type: ignore
            _coll_type="EPISODE",
            _hyper_param=HyperParamRDPG(),
            _truck=truck,
            _driver=driver,
            _pool_key=args.pool_key,
            _data_folder=str(data_root),
            _infer_mode=args.infer_mode,
            _resume=args.resume,
        )

    try:
        avatar = Avatar(
            _truck=truck,
            _driver=driver,
            _agent=agent,
            cloud=args.cloud,
            ui=args.ui,
            resume=args.resume,
            infer_mode=args.infer_mode,
            record=args.record_table,
            data_root=data_root,
            logger=logger,
        )
    except TypeError as e:
        logger.error(
            f"{{'header': 'Project Exception TypeError', " f"'exception': '{e}'}}",
            extra=dictLogger,
        )
        sys.exit(1)
    except Exception as e:
        logger.error(
            f"{{'header': 'main Exception', " f"'exception': '{e}'}}",
            extra=dictLogger,
        )
        sys.exit(1)

    if args.infer_mode:
        main_run = avator.infer
    else:
        main_run = avatar.train

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        executor.submit(
            avatar.kvaser.produce,  # observe thread
        )
        executor.submit(
            avatar.kvaser.countdown,  # countdown thread
            [
                avatar.interrupt_event,
                avatar.exit_event,
            ],
        )
        executor.submit(
            avatar.cruncher.crunch,  # data crunch thread
            [
                avatar.start_event,
                avatar.stop_event,
                avatar.interrupt_event,
                avatar.exit_event,
            ],
        )
        executor.submit(
            avatar.kvaser.consume,  # flash thread
            [
                avatar.start_event,
                avatar.stop_event,
                avatar.interrupt_event,
                avatar.exit_event,
            ],
        )

        # default behavior is observe will start and send out all the events to orchestrate other three threads.
