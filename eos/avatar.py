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

import abc
import argparse
import concurrent.futures

# logging
import logging

# system imports
import os
import sys

# third party imports
from dataclasses import dataclass
from logging.handlers import SocketHandler
from pathlib import Path, PurePosixPath
from threading import Event
from typing import Optional, Union, cast

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd  # type: ignore

# tf.debugging.set_log_device_placement(True)
# visualization import
import tensorflow as tf
from git import Repo
from pythonjsonlogger import jsonlogger  # type: ignore
from tensorflow.summary import SummaryWriter, create_file_writer, scalar  # type: ignore
from typeguard import check_type  # type: ignore

from eos import proj_root
from eos.agent import DDPG, DPG, RDPG
from eos.agent.utils import HyperParamDDPG, HyperParamRDPG
from eos.data_io.config import (
    TruckInField,
    TruckInCloud,
    Driver,
    CANMessenger,
    TripMessenger,
    str_to_can_server,
    str_to_driver,
    str_to_trip_server,
    str_to_truck,
)
from eos.data_io.dataflow import (
    Cloud,
    Cruncher,
    Kvaser,
    Pipeline,
)
from eos.data_io.utils import dictLogger, logger, GracefulKiller

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# system warnings and numpy warnings handling
# np.warnings.filterwarnings('ignore', category=DeprecationWarning)


@dataclass(kw_only=True)
class Avatar(abc.ABC):
    _truck: Union[TruckInField, TruckInCloud]
    _driver: Driver
    _can_server: CANMessenger
    _trip_server: Optional[TripMessenger]
    _agent: DPG  # set by derived Avatar like AvatarDDPG
    logger: logging.Logger
    dictLogger: dict
    vehicle_interface: Union[Kvaser, Cloud] = None
    _resume: bool = True
    _infer_mode: bool = False
    cruncher: Optional[Cruncher] = None
    data_root: Path = Path(".") / "data"
    log_root: Optional[Path] = None
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

        tf.keras.backend.set_floatx("float32")
        self.logger_control_flow.info(
            f"{{'header': 'tensorflow device lib:\n{tf.config.list_physical_devices()}'}}",
            extra=self.dictLogger,
        )
        self.logger_control_flow.info(
            f"{{'header': 'Tensorflow Imported!'}}", extra=self.dictLogger
        )

        if self.can_server.protocol == "udp":
            self.vehicle_interface: Kvaser = Kvaser(  # Producer~Consumer~Filter
                truck=cast(TruckInField, self.truck),
                driver=self.driver,
                can_server=self.can_server,
                resume=self.resume,
                data_dir=self.data_root,
                logger=self.logger,
                dictLogger=self.dictLogger,
            )
        else:  # self.can_server.protocol == 'tcp'
            self.vehicle_interface: Cloud = Cloud(  # Producer~Consumer
                truck=cast(TruckInCloud, self.truck),
                driver=self.driver,
                can_server=self.can_server,
                trip_server=self.trip_server,
                resume=self.resume,
                data_dir=self.data_root,
                logger=self.logger,
                dictLogger=self.dictLogger,
            )

        self.cruncher = Cruncher(  # Consumer
            agent=self.agent,
            truck=self.truck,
            driver=self.driver,
            resume=self.resume,
            infer_mode=self.infer_mode,
            data_dir=self.data_root,
            logger=self.logger,
            dictLogger=self.dictLogger,
        )

    @property
    def agent(self) -> Optional[DPG]:
        return self._agent

    @agent.setter
    def agent(self, value: DPG) -> None:
        self._agent = value

    @property
    def truck(self) -> Union[TruckInField, TruckInCloud]:
        return self._truck

    @truck.setter
    def truck(self, value: Union[TruckInField, TruckInCloud]) -> None:
        self._truck = value

    @property
    def driver(self) -> Driver:
        return self._driver

    @driver.setter
    def driver(self, value: Driver) -> None:
        self._driver = value

    @property
    def can_server(self) -> CANMessenger:
        return self._can_server

    @can_server.setter
    def can_server(self, value: CANMessenger) -> None:
        self._can_server = value

    @property
    def trip_server(self) -> TripMessenger:
        return self._trip_server

    @trip_server.setter
    def trip_server(self, value: TripMessenger) -> None:
        self._trip_server = value

    @property
    def resume(self) -> bool:
        return self._resume

    @resume.setter
    def resume(self, value: bool) -> None:
        self._resume = value

    @property
    def infer_mode(self) -> bool:
        return self._infer_mode

    @infer_mode.setter
    def infer_mode(self, value: bool) -> None:
        self._infer_mode = value

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
        "-i",
        "--interface",
        type=str,
        default="can_udp_svc",
        help="url for remote can server, e.g. 10.10.0.6:30865, or name, e.g. can_cloud, can_intra, can_udp_svc",
    )
    parser.add_argument(
        "-t",
        "--trip",
        type=str,
        default="local_udp",
        help="trip messenger, url or name, e.g. rocket_cloud, local_udp",
    )
    parser.add_argument(
        "-c",
        "--control",
        type=str,
        default="UDP",
        help="HMI Control Interface: "
        "'RMQ' for mobile phone (using rocketmq for training/assessment); "
        "'UDP' for local hmi (using loopback tcp for training/assessment); "
        "'DUMMY' for non-interaction for inference only and testing purpose",
    )
    parser.add_argument(
        "-a",
        "--agent",
        type=str,
        default="ddpg",
        help="RL agent choice: 'ddpg' for DDPG; 'rdpg' for Recurrent DPG",
    )

    parser.add_argument(
        "-r",
        "--resume",
        default=True,
        help="resume the last training with restored model, checkpoint and pedal map",
        action="store_true",
    )

    parser.add_argument(
        "-l",
        "--learning",
        default=True,
        help="True for learning , with model update and training. False for inference only",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default=".",
        help="relative path to be saved, for create sub-folder for different drivers",
    )
    parser.add_argument(
        "-o",
        "--output",
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
        truck: Union[TruckInField, TruckInCloud] = str_to_truck(args.vehicle)
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
        can_server = str_to_can_server(args.interface)
    except KeyError:
        raise KeyError(f"can server {args.interface} not found in config file")
    else:
        logger.info(f"CAN Server found: {can_server.server_name}", extra=dictLogger)

    try:
        trip_server = str_to_trip_server(args.trip)
    except KeyError:
        raise KeyError(f"trip server {args.web} not found in config file")
    else:
        logger.info(f"Trip Server found: {trip_server.server_name}", extra=dictLogger)

    assert args.agent in ["ddpg", "rdpg"], "agent must be either ddpg or rdpg"

    if args.resume:
        data_root = proj_root.joinpath("data/" + truck.vin + "-" + driver.pid).joinpath(
            args.path
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
            _pool_key=args.output,
            _data_folder=str(data_root),
            _infer_mode=(not args.learning),
            _resume=args.resume,
        )
    else:  # args.agent == 'rdpg':
        agent: RDPG = RDPG(  # type: ignore
            _coll_type="EPISODE",
            _hyper_param=HyperParamRDPG(),
            _truck=truck,
            _driver=driver,
            _pool_key=args.output,
            _data_folder=str(data_root),
            _infer_mode=(not args.learning),
            _resume=args.resume,
        )

    try:
        avatar = Avatar(
            _truck=truck,
            _driver=driver,
            _agent=agent,
            _can_server=can_server,
            _trip_server=trip_server,
            logger=logger,
            dictLogger=dictLogger,
            _resume=args.resume,
            _infer_mode=(not args.learning),
            data_root=data_root,
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

    # initialize dataflow: pipelines, sync events among the threads
    observe_pipeline = Pipeline[pd.DataFrame](
        maxsize=3
    )  # pipeline for observations (type dataframe)
    flash_pipeline = Pipeline[pd.DataFrame](
        maxsize=3
    )  # pipeline for flashing torque tables (type dataframe)
    start_event = Event()
    stop_event = Event()
    interrupt_event = Event()
    exit_event = Event()
    flash_event = Event()

    # Gracefulkiller instance can be created only in the main thread!
    killer = GracefulKiller(exit_event)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=2, thread_name_prefix='Avatar'
    ) as executor:
        executor.submit(
            avatar.vehicle_interface.ignite,  # observe thread (spawns 4 threads for input, HMI and output)
            observe_pipeline,  # input port; output
            flash_pipeline,  # out port; input
            start_event,
            stop_event,
            interrupt_event,
            flash_event,
            exit_event,
        )

        executor.submit(
            avatar.cruncher.filter,  # data crunch thread
            observe_pipeline,  # output port; input
            flash_pipeline,  # input port; output
            start_event,
            stop_event,
            interrupt_event,
            flash_event,
            exit_event,
        )

    # default behavior is "observe" will start and send out all the events to orchestrate other three threads.
    logger.info("Program exit!")
