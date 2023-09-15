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
import json

# logging
import logging
import math

# system imports
import os
import sys
import threading
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

from eos.data_io.dataflow import EcuPipeline, CloudPipeline

from eos.visualization import plot_3d_figure, plot_to_image


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# system warnings and numpy warnings handling
warnings.filterwarnings("ignore", message="currentThread", category=DeprecationWarning)
# np.warnings.filterwarnings('ignore', category=DeprecationWarning)


@dataclass(kw_only=True)
class Avatar(abc.ABC):
    truck: Truck
    driver: Driver
    _agent: DPG  # set by derived Avatar like AvatarDDPG
    pipeline: EcuPipeline
    cloud: bool  # determined by truck type
    ui: str
    logger: logging.Logger
    resume: bool = True
    infer_mode: bool = False
    record: bool = True
    data_root: Path = Path(".") / "data"
    table_root: Path = Path(".") / "tables"
    program_start: bool = False
    program_exit: bool = False
    train_summary_writer: Optional[SummaryWriter] = None
    log_root: Path = Path(".") / "py_log"
    logger_control_flow: Optional[logging.Logger] = None
    tflog: Optional[logging.Logger] = None
    vcu_calib_table0: Optional[pd.DataFrame] = None  # initial calibration table
    vcu_calib_table1: Optional[
        pd.DataFrame
    ] = None  # dynamic calibration table, updated by agent

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

        if type(self.truck) == TruckInCloud:
            self.cloud = True
        else:
            self.cloud = False

        self.dictLogger = dictLogger
        # self.dictLogger = {"user": inspect.currentframe().f_code.co_name}

        self.set_logger()
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

        if self.cloud:
            self.pipeline = CloudPipeline()
            # reset proxy (internal site force no proxy)
        else:
            self.pipeline = EcuPipeline()

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
        self.init_threads_data()
        self.logger_control_flow.info(
            f"{{'header': 'Thread data Initialization done!'}}",
            extra=self.dictLogger,
        )

    def start_threads(self) -> None:
        self.evt_epi_done = Event()
        self.evt_remote_get = Event()
        self.evt_remote_flash = Event()

        self.evt_epi_done.clear()
        self.evt_remote_flash.clear()  # initially false, explicitly set the remote flash event to 'False' to start with
        self.thr_countdown = Thread(
            target=self.capture_countdown_handler,
            name="countdown",
            args=[self.evt_epi_done, self.evt_remote_get, self.evt_remote_flash],
        )
        self.thr_countdown.start()

        if self.cloud:
            if self.ui == "RMQ":
                self.logger.info(f"{{'header': 'Use phone UI'}}", extra=self.dictLogger)
                self.thr_observe = Thread(
                    target=self.remote_hmi_rmq_state_machine,
                    name="observe",
                    args=[
                        self.evt_epi_done,
                        self.evt_remote_get,
                        self.evt_remote_flash,
                    ],
                )
                self.thr_observe.start()
            elif self.ui == "TCP":
                self.logger.info(f"{{'header': 'Use local UI'}}", extra=self.dictLogger)
                self.thr_observe = Thread(
                    target=self.remote_hmi_tcp_state_machine,
                    name="observe",
                    args=[
                        self.evt_epi_done,
                        self.evt_remote_get,
                        self.evt_remote_flash,
                    ],
                )
                self.thr_observe.start()
            elif self.ui == "NUMB":
                self.logger.info(f"{{'header': 'Use cloud UI'}}", extra=self.dictLogger)
                self.thr_observe = Thread(
                    target=self.remote_hmi_no_state_machine,
                    name="observe",
                    args=[
                        self.evt_epi_done,
                        self.evt_remote_get,
                        self.evt_remote_flash,
                    ],
                )
                self.thr_observe.start()
            else:
                raise ValueError("Unknown HMI type")

            self.thr_remote_get = Thread(
                target=self.remote_get_handler,
                name="remoteget",
                args=[self.evt_remote_get, self.evt_remote_flash],
            )
            self.thr_remote_get.start()

            self.thr_flash = Thread(
                target=self.remote_flash_vcu, name="flash", args=[self.evt_remote_flash]
            )
            self.thr_flash.start()
        else:
            self.thr_observe = Thread(
                target=self.kvaser_get_truck_status,
                name="observe",
                args=[self.evt_epi_done],
            )
            self.thr_observe.start()
            self.thr_flash = Thread(
                target=self.kvaser_flash_vcu, name="flash", args=[self.evt_remote_flash]
            )
            self.thr_flash.start()

    @property
    def agent(self) -> Union[DPG, None]:
        return self._agent

    @agent.setter
    def agent(self, value: DPG) -> None:
        self._agent = value

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

        # TQD_trqTrqSetECO_MAP_v

    # tracer.start()

    def train(self):
        # Start thread for flashing vcu, flash first
        evt_epi_done = threading.Event()
        evt_remote_get = threading.Event()
        evt_remote_flash = threading.Event()

        """
        ## train
        """
        running_reward = 0.0
        th_exit = False
        epi_cnt_local = 0

        # Gracefulkiller only in the main thread!
        killer = GracefulKiller()

        self.logger_control_flow.info(
            f"main Initialization done!", extra=self.dictLogger
        )
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
                # self.logger.info(f'Wait for start!', extra=self.dictLogger)
                continue

            # tf.summary.trace_on(graph=True, profiler=True)

            self.logger_control_flow.info(
                "----------------------", extra=self.dictLogger
            )
            self.logger_control_flow.info(
                f"{{'header': 'episode starts!', " f"'episode': {epi_cnt}}}",
                extra=self.dictLogger,
            )
            # mongodb default to UTC time

            # Get the initial motion_power data for the initial quadruple (s, a, r, s')_{-1}
            while True:
                motion_power = None
                with self.hmi_lock:  # wait for tester to kick off or to exit
                    th_exit = self.program_exit  # if program_exit is False,

                if th_exit:
                    self.logger_control_flow.info(
                        f"{{'header': 'Program exit!!!', ",
                        extra=self.dictLogger,
                    )
                    break

                try:
                    motion_power = check_type(self.motion_power_queue, Queue).get(
                        block=True, timeout=10
                    )
                    # check_type(self.motion_power_queue, Queue).task_done()
                    break  # break the while loop if we get the first data
                except TimeoutError:
                    self.logger_control_flow.info(
                        f"{{'header': 'No data in the input Queue, Timeout!!!', "
                        f"'episode': {epi_cnt}}}",
                        extra=self.dictLogger,
                    )
                    continue
                except queue.Empty:
                    self.logger_control_flow.info(
                        f"{{'header': 'No data in the input Queue, Empty!!!', "
                        f"'episode': {epi_cnt}}}",
                        extra=self.dictLogger,
                    )
                    continue

            if th_exit:
                continue

            self.agent.start_episode(pd.Timestamp.now(tz=self.truck.tz))
            step_count = 0
            episode_reward = 0.0
            prev_timestamp = self.agent.episode_start_dt
            check_type(motion_power, pd.DataFrame)
            prev_state = assemble_state_ser(
                motion_power.loc[:, ["timestep", "velocity", "thrust", "brake"]],
                tz=self.truck.tz,
            )  # s_{-1}
            zero_torque_map_line = np.zeros(
                shape=(1, 1, self.truck.torque_flash_numel),  # [1, 1, 4*17]
                dtype=np.float32,
            )  # first zero last_actions is a 3D tensor
            prev_action = assemble_action_ser(
                torque_map_line=zero_torque_map_line,
                torque_table_row_names=self.agent.torque_table_row_names,
                table_start=0,
                flash_start_ts=pd.to_datetime(prev_timestamp),
                flash_end_ts=pd.Timestamp.now(self.truck.tz),
                torque_table_row_num_flash=self.truck.torque_table_row_num_flash,
                torque_table_col_num=self.truck.torque_table_col_num,
                speed_scale=self.truck.speed_scale,
                pedal_scale=self.truck.pedal_scale,
                tz=self.truck.tz
            )  # a_{-1}
            step_reward = 0.0
            # reward is measured in next step

            self.logger_control_flow.info(
                f"{{'header': 'episode init done!', " f"'episode': {epi_cnt}}}",
                extra=self.dictLogger,
            )
            self.evt_remote_flash.set()  # kick off the episode capturing
            b_flashed = False
            tf.debugging.set_log_device_placement(True)
            with tf.device("/GPU:0"):
                while (
                    not epi_end
                ):  # end signal, either the round ends normally or user interrupt
                    if killer.kill_now:
                        self.logger_control_flow.info(f"Process is being killed!!!")
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

                    motion_power_queue_size = check_type(
                        self.motion_power_queue, Queue
                    ).qsize()
                    self.logger_control_flow.info(
                        f" motion_power_queue.qsize(): {motion_power_queue_size}"
                    )
                    if epi_end and done and (motion_power_queue_size > 2):
                        # self.logc.info(f"motion_power_queue.qsize(): {self.motion_power_queue.qsize()}")
                        self.logger_control_flow.info(
                            f"{{'header': 'Residue in Queue is a sign of disordered sequence, interrupted!'}}"
                        )
                        done = (
                            False  # this local done is true done with data exploitation
                        )

                    if epi_end:  # stop observing
                        # g and inferring
                        continue

                    try:
                        # self.logc.info(
                        #     f"E{epi_cnt} Wait for an object!!!", extra=self.dictLogger
                        # )

                        motion_power = check_type(self.motion_power_queue, Queue).get(
                            block=True, timeout=1.55
                        )
                    except TimeoutError:
                        self.logger_control_flow.info(
                            f"{{'header': 'No data in the input Queue Timeout!!!', "
                            f"'episode': {epi_cnt}}}",
                            extra=self.dictLogger,
                        )
                        continue
                    except queue.Empty:
                        self.logger_control_flow.info(
                            f"{{'header': 'No data in the input Queue empty Queue!!!', "
                            f"'episode': {epi_cnt}}}",
                            extra=self.dictLogger,
                        )
                        continue

                    self.logger_control_flow.info(
                        f"{{'header': 'start', "
                        f"'step': {step_count}, "
                        f"'episode': {epi_cnt}}}",
                        extra=self.dictLogger,
                    )  # env.step(action) action is flash the vcu calibration table

                    # !!!no parallel even!!!
                    # predict action probabilities and estimated future rewards
                    # from environment state
                    # for causal rl, the odd indexed observation/reward are caused by last action
                    # skip the odd indexed observation/reward for policy to make it causal

                    # assemble state
                    timestamp: pd.Timestamp = motion_power.loc[
                        0, "timestep"
                    ]  # only take the first timestamp, as frequency is fixed at 50Hz, the rest is saved in another col

                    # motion_power.loc[:, ['timestep', 'velocity', 'thrust', 'brake']]
                    state = assemble_state_ser(
                        motion_power.loc[:, ["timestep", "velocity", "thrust", "brake"]],
                        tz=self.truck.tz,
                    )

                    # assemble reward, actually the reward from last action
                    # pow_t = motion_power.loc[:, ['current', 'voltage']]
                    reward = assemble_reward_ser(
                        motion_power.loc[:, ["current", "voltage"]],
                        self.truck.observation_sampling_rate,
                        ts=pd.Timestamp.now(tz=self.truck.tz),
                    )
                    work = reward[("work", 0)]
                    episode_reward += float(work)

                    self.logger_control_flow.info(
                        f"{{'header': 'assembling state and reward!', "
                        f"'episode': {epi_cnt}}}",
                        extra=self.dictLogger,
                    )

                    #  separate the inference and flash in order to avoid the action change incurred reward noise
                    if b_flashed is False:  # the active half step
                        #  at step 0: [ep_start, None (use zeros), a=0, r=0, s=s_0]
                        #  at step n: [t=t_{n-1}, s=s_{n-1}, a=a_{n-1}, r=r_{n-1}, s'=s_n]
                        #  at step N: [t=t_{N-1}, s=s_{N-1}, a=a_{N-1}, r=r_{N-1}, s'=s_N]
                        reward[("work", 0)] = (
                            work + step_reward
                        )  # reward is the sum of flashed and not flashed step
                        self.agent.deposit(
                            prev_timestamp,
                            prev_state,
                            prev_action,
                            reward,  # reward from last action
                            state,
                        )  # (s_{-1}, a_{-1}, r_{-1}, s_0), (s_0, a_0, r_0, s_1), ..., (s_{N-1}, a_{N-1}, r_{N-1}, s_N)

                        # Inference !!!
                        # stripping timestamps from state, (later flatten and convert to tensor)
                        # agent return the inferred action sequence without batch and time dimension
                        torque_map_line = self.agent.actor_predict(
                            state[["velocity", "thrust", "brake"]]
                        )  # model input requires fixed order velocity col -> thrust col -> brake col
                        #  !!! training with samples of the same order!!!

                        self.logger_control_flow.info(
                            f"{{'header': 'inference done with reduced action space!', "
                            f"'episode': {epi_cnt}}}",
                            extra=self.dictLogger,
                        )
                        # flash the vcu calibration table and assemble action
                        flash_start_ts = pd.Timestamp.now(self.truck.tz)
                        self.tableQueue.put(torque_map_line)
                        self.logger_control_flow.info(
                            f"{{'header': 'Action Push table', "
                            f"'StartIndex': {table_start}, "
                            f"'qsize': {self.tableQueue.qsize()}}}",
                            extra=self.dictLogger,
                        )

                        # wait for remote flash to finish
                        # with check_type(self.flash_env_lock, Lock):
                        self.evt_remote_flash.clear()
                        self.evt_remote_flash.wait()
                        self.logger_control_flow.info(
                            f"{{'header': 'after flash lock wait!",
                            extra=self.dictLogger,
                        )
                        flash_end_ts = pd.Timestamp.now(self.truck.tz)

                        action = assemble_action_ser(
                            torque_map_line,
                            self.agent.torque_table_row_names,
                            table_start,
                            flash_start_ts,
                            flash_end_ts,
                            self.truck.torque_table_row_num_flash,
                            self.truck.torque_table_col_num,
                            self.truck.speed_scale,
                            self.truck.pedal_scale,
                            self.truck.tz
                        )

                        prev_timestamp = timestamp
                        prev_state = state
                        prev_action = action
                        b_flashed = True
                    else:  # if bFlashed is True, the dummy half step
                        step_reward = float(
                            work
                        )  # reward from the step without flashing action
                        b_flashed = False

                    # TODO add speed sum as positive reward
                    self.logger_control_flow.info(
                        f"{{'header': 'Step done',"
                        f"'step': {step_count}, "
                        f"'episode': {epi_cnt}}}",
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
                self.logger_control_flow.info(
                    f"{{'header': 'interrupted, waits for next episode to kick off!' "
                    f"'episode': {epi_cnt}}}",
                    extra=self.dictLogger,
                )
                # send ready signal to trip server
                if self.ui == "mobile":
                    ret = self.rmq_producer.send_sync(self.rmq_message_ready)
                    self.logger_control_flow.info(
                        f"{{'header': 'Sending ready signal to trip server', "
                        f"'status': '{ret.status}', "
                        f"'msg-id': '{ret.msg_id}', "
                        f"'offset': '{ret.offset}'}}",
                        extra=self.dictLogger,
                    )
                continue  # otherwise assuming the history is valid and back propagate

            self.agent.end_episode()  # deposit history

            self.logger_control_flow.info(
                f"{{'header': 'Episode end.', "
                f"'episode': '{epi_cnt}', "
                f"'timestamp': '{datetime.now(self.truck.tz)}'}}",
                extra=self.dictLogger,
            )

            critic_loss = 0
            actor_loss = 0

            self.logger_control_flow.info(
                "{{'header': 'Learning and updating 6 times!'}}"
            )
            for k in range(6):
                # self.logger.info(f"BP{k} starts.", extra=self.dictLogger)
                if self.agent.buffer.pool.cnt > 0:
                    for k in range(6):
                        (critic_loss, actor_loss) = self.agent.train()
                        self.agent.soft_update_target()
                else:
                    self.logger_control_flow.info(
                        f"{{'header': 'Buffer empty, no learning!'}}",
                        extra=self.dictLogger,
                    )
                    self.logger_control_flow.info(
                        "++++++++++++++++++++++++", extra=self.dictLogger
                    )
                    break
            # Checkpoint manager save model
            self.agent.save_ckpt()

            self.logger_control_flow.info(
                f"{{'header': 'losses after 6 times BP', "
                f"'episode': {epi_cnt}, "
                f"'critic loss': {critic_loss}, "
                f"'actor loss': {actor_loss}}}",
                extra=self.dictLogger,
            )

            # update running reward to check condition for solving
            running_reward = 0.05 * (-episode_reward) + (1 - 0.05) * running_reward

            # Create a matplotlib 3d figure, //export and save in log
            fig = plot_3d_figure(self.vcu_calib_table1)

            # tf logging after episode ends
            # use local episode counter epi_cnt_local tf.summary.writer;
            # otherwise specify multiple self.logdir and automatic switch
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
                #     name="veos_trace", step=epi_cnt_local, profiler_out_dir=train_log_dir
                # )

            epi_cnt_local += 1
            plt.close(fig)

            self.logger_control_flow.info(
                f"{{'episode': {epi_cnt}, " f"'reward': {episode_reward}}}",
                extra=self.dictLogger,
            )

            self.logger_control_flow.info(
                "----------------------", extra=self.dictLogger
            )
            if epi_cnt % 10 == 0:
                self.logger_control_flow.info(
                    "++++++++++++++++++++++++", extra=self.dictLogger
                )
                self.logger_control_flow.info(
                    f"{{'header': 'Running reward': {running_reward:.2f}, "
                    f"'episode': '{epi_cnt}'}}",
                    extra=self.dictLogger,
                )
                self.logger_control_flow.info(
                    "++++++++++++++++++++++++", extra=self.dictLogger
                )

            # send ready signal to trip server
            if self.ui == "mobile":
                ret = self.rmq_producer.send_sync(self.rmq_message_ready)
                self.logger.info(
                    f"{{'header': 'Sending ready signal to trip server', "
                    f"'status': '{ret.status}', "
                    f"'msg_id': '{ret.msg_id}', "
                    f"'offset': '{ret.offset}'}}",
                    extra=self.dictLogger,
                )
        # TODO terminate condition to be defined: reward > limit (percentage); time too long
        # with self.train_summary_writer.as_default():
        #     tf.summary.trace_export(
        #         name="veos_trace",
        #         step=epi_cnt_local,
        #         profiler_out_dir=self.train_log_dir,
        #     )
        self.thr_observe.join()
        if self.cloud:
            self.thr_remote_get.join()
        self.thr_flash.join()
        self.thr_countdown.join()

        self.logger_control_flow.info(
            f"{{'header': 'main dies!!!!'}}", extra=self.dictLogger
        )

    def infer(self):
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
            self.thr_remote_get = Thread(
                target=self.remote_get_handler,
                name="remoteget",
                args=[evt_remote_get, evt_remote_flash],
            )
            self.thr_remote_get.start()

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

        self.logger_control_flow.info(
            f"main Initialization done!", extra=self.dictLogger
        )
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
                # self.logger.info(f'Wait for start!', extra=self.dictLogger)
                continue

            # tf.summary.trace_on(graph=True, profiler=True)

            self.logger_control_flow.info(
                "----------------------", extra=self.dictLogger
            )
            self.logger_control_flow.info(
                f"{{'header': 'episode starts!', " f"'episode': {epi_cnt}}}",
                extra=self.dictLogger,
            )

            # mongodb default to UTC time

            # Get the initial motion_power data for the initial quadruple (s, a, r, s')_{-1}
            while True:
                motion_power = None
                try:
                    motion_power = check_type(self.motion_power_queue, Queue).get(
                        block=True, timeout=1.55
                    )
                    # check_type(self.motion_power_queue, Queue).task_done()
                    break  # break the while loop if we get the first data
                except TimeoutError:
                    self.logger_control_flow.info(
                        f"{{'header': 'No data in the input Queue, Timeout!!!', "
                        f"'episode': {epi_cnt}}}",
                        extra=self.dictLogger,
                    )
                    continue
                except queue.Empty:
                    self.logger_control_flow.info(
                        f"{{'header': 'No data in the input Queue, Empty!!!', "
                        f"'episode': {epi_cnt}}}",
                        extra=self.dictLogger,
                    )
                    continue

            self.agent.start_episode(datetime.now())
            step_count = 0
            episode_reward = 0
            prev_timestamp = self.agent.episode_start_dt
            check_type(motion_power, pd.DataFrame)
            prev_state = assemble_state_ser(
                motion_power.loc[:, ["timestep", "velocity", "thrust", "brake"]]
            )  # s_{-1}
            zero_torque_map_line = np.zeros(
                shape=(1, 1, self.truck.torque_flash_numel),  # [1, 1, 4*17]
                dtype=tf.float32,
            )  # first zero last_actions is a 3D tensor
            prev_action = assemble_action_ser(
                zero_torque_map_line,
                self.agent.torque_table_row_names,
                table_start,
                flash_start_ts,
                flash_end_ts,
                self.truck.torque_table_row_num_flash,
                self.truck.torque_table_col_num,
                self.truck.speed_scale,
                self.truck.pedal_scale,
            )  # a_{-1}
            # reward is measured in next step

            self.logger_control_flow.info(
                f"{{'header': 'start', "
                f"'step': {step_count}, "
                f"'episode': {epi_cnt}}}",
                extra=self.dictLogger,
            )
            tf.debugging.set_log_device_placement(True)
            with tf.device("/GPU:0"):
                while (
                    not epi_end
                ):  # end signal, either the round ends normally or user interrupt
                    if killer.kill_now:
                        self.logger_control_flow.info(f"Process is being killed!!!")
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

                    motion_power_queue_size = check_type(
                        self.motion_power_queue, Queue
                    ).qsize()
                    self.logger_control_flow.info(
                        f"motion_power_queue.qsize(): {motion_power_queue_size}"
                    )
                    if epi_end and done and (motion_power_queue_size > 2):
                        # self.logc.info(f"motion_power_queue.qsize(): {self.motion_power_queue.qsize()}")
                        self.logger_control_flow.info(
                            f"{{'header': 'Residue in Queue is a sign of disordered sequence, interrupted!'}}"
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

                        motion_power = check_type(self.motion_power_queue, Queue).get(
                            block=True, timeout=1.55
                        )
                    except TimeoutError:
                        self.logger_control_flow.info(
                            f"{{'header': 'No data in the input Queue!!!', "
                            f"'episode': {epi_cnt}}}",
                            extra=self.dictLogger,
                        )
                        continue

                    self.logger_control_flow.info(
                        f"{{'header': 'start', "
                        f"'step': {step_count}, "
                        f"'episode': {epi_cnt}}}",
                        extra=self.dictLogger,
                    )  # env.step(action) action is flash the vcu calibration table

                    # !!!no parallel even!!!
                    # predict action probabilities and estimated future rewards
                    # from environment state
                    # for causal rl, the odd indexed observation/reward are caused by last action
                    # skip the odd indexed observation/reward for policy to make it causal

                    # assemble state
                    timestamp = motion_power.loc[
                        0, "timestep"
                    ]  # only take the first timestamp, as frequency is fixed at 50Hz, the rest is saved in another col

                    # motion_power.loc[:, ['timestep', 'velocity', 'thrust', 'brake']]
                    state = assemble_state_ser(
                        motion_power.loc[:, ["timestep", "velocity", "thrust", "brake"]]
                    )

                    # assemble reward, actually the reward from last action
                    # pow_t = motion_power.loc[:, ['current', 'voltage']]
                    reward = assemble_reward_ser(
                        motion_power.loc[:, ["current", "voltage"]],
                        self.truck.observation_sampling_rate,
                    )
                    work = reward[("work", 0)]
                    episode_reward += work

                    self.logger_control_flow.info(
                        f"{{'header': 'assembling state and reward!', "
                        f"'episode': {epi_cnt}}}",
                        extra=self.dictLogger,
                    )

                    #  at step 0: [ep_start, None (use zeros), a=0, r=0, s=s_0]
                    #  at step n: [t=t_{n-1}, s=s_{n-1}, a=a_{n-1}, r=r_{n-1}, s'=s_n]
                    #  at step N: [t=t_{N-1}, s=s_{N-1}, a=a_{N-1}, r=r_{N-1}, s'=s_N]
                    self.agent.deposit(
                        prev_timestamp,
                        prev_state,
                        prev_action,
                        reward,  # reward from last action
                        state,
                    )  # (s_{-1}, a_{-1}, r_{-1}, s_0), (s_0, a_0, r_0, s_1), ..., (s_{N-1}, a_{N-1}, r_{N-1}, s_N)

                    # Inference !!!
                    # stripping timestamps from state, (later flatten and convert to tensor)
                    # agent return the inferred action sequence without batch and time dimension
                    torque_map_line = self.agent.actor_predict(
                        state[["velocity", "thrust", "brake"]]
                    )  # model input requires fixed order velocity col -> thrust col -> brake col
                    #  !!! training with samples of the same order!!!

                    self.logger_control_flow.info(
                        f"{{'header': 'inference done with reduced action space!', "
                        f"'episode': {epi_cnt}}}",
                        extra=self.dictLogger,
                    )
                    # flash the vcu calibration table and assemble action
                    flash_start_ts = pd.to_datetime(datetime.now())
                    self.tableQueue.put(torque_map_line)
                    self.logger_control_flow.info(
                        f"{{'header': 'Action Push table', "
                        f"'StartIndex': {table_start}, "
                        f"'qsize': {self.tableQueue.qsize()}}}",
                        extra=self.dictLogger,
                    )

                    # wait for remote flash to finish
                    evt_remote_flash.wait()
                    flash_end_ts = pd.to_datetime(datetime.now())

                    action = assemble_action_ser(
                        torque_map_line.to_numpy(),
                        self.agent.torque_table_row_names,
                        table_start,
                        flash_start_ts,
                        flash_end_ts,
                        self.truck.torque_table_row_num_flash,
                        self.truck.torque_table_col_num,
                        self.truck.speed_scale,
                        self.truck.pedal_scale,
                    )

                    prev_timestamp = timestamp
                    prev_state = state
                    prev_action = action

                    # TODO add speed sum as positive reward
                    self.logger_control_flow.info(
                        f"{{'header': 'Step done',"
                        f"'step': {step_count}, "
                        f"'episode': {epi_cnt}}}",
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
                self.logger_control_flow.info(
                    f"{{'header': 'interrupted, waits for next episode to kick off!' "
                    f"'episode': {epi_cnt}}}",
                    extra=self.dictLogger,
                )
                # send ready signal to trip server
                if self.ui == "mobile":
                    ret = self.rmq_producer.send_sync(self.rmq_message_ready)
                    self.logger_control_flow.info(
                        f"{{'header': 'Sending ready signal to trip server', "
                        f"'status': '{ret.status}', "
                        f"'msg-id': '{ret.msg_id}', "
                        f"'offset': '{ret.offset}'}}",
                        extra=self.dictLogger,
                    )
                continue  # otherwise assuming the history is valid and back propagate

            self.agent.end_episode()  # deposit history

            self.logger.info(
                f"{{'header': 'Episode end.', " f"'episode': {epi_cnt}, ",
                f"'timestamp': {datetime.now()}}}",
                extra=self.dictLogger,
            )

            critic_loss = 0
            actor_loss = 0
            (critic_loss, actor_loss) = self.agent.get_losses()
            # FIXME bugs in maximal sequence length for ungraceful testing
            # self.logc.info("Nothing to be done for rdpg!")
            self.logger_control_flow.info(
                "{{'header': 'No Learning, just calculating loss.'}}"
            )

            self.logger_control_flow.info(
                f"{{'header': 'losses after 6 times BP', "
                f"'episode': {epi_cnt}, "
                f"'critic loss': {critic_loss}, "
                f"'actor loss': {actor_loss}}}",
                extra=self.dictLogger,
            )

            # update running reward to check condition for solving
            running_reward = 0.05 * (-episode_reward) + (1 - 0.05) * running_reward

            # Create a matplotlib 3d figure, //export and save in log
            fig = plot_3d_figure(self.vcu_calib_table1)

            # tf logging after episode ends
            # use local episode counter epi_cnt_local tf.summary.writer;
            # otherwise specify multiple self.logdir and automatic switch
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
                #     name="veos_trace", step=epi_cnt_local, profiler_out_dir=train_log_dir
                # )

            epi_cnt_local += 1
            plt.close(fig)

            self.logger_control_flow.info(
                f"{{'episode': {epi_cnt}, " f"'reward': {episode_reward}}}",
                extra=self.dictLogger,
            )

            self.logger_control_flow.info(
                "----------------------", extra=self.dictLogger
            )
            if epi_cnt % 10 == 0:
                self.logger_control_flow.info(
                    "++++++++++++++++++++++++", extra=self.dictLogger
                )
                self.logger_control_flow.info(
                    f"{{'header': 'Running reward': {running_reward:.2f}, "
                    f"'episode': '{epi_cnt}'}}",
                    extra=self.dictLogger,
                )
                self.logger_control_flow.info(
                    "++++++++++++++++++++++++", extra=self.dictLogger
                )

            # send ready signal to trip server
            if self.ui == "mobile":
                ret = self.rmq_producer.send_sync(self.rmq_message_ready)
                self.logger.info(
                    f"{{'header': 'Sending ready signal to trip server', "
                    f"'status': '{ret.status}', "
                    f"'msg_id': '{ret.msg_id}', "
                    f"'offset': '{ret.offset}'}}",
                    extra=self.dictLogger,
                )
        # TODO terminate condition to be defined: reward > limit (percentage); time too long
        # with self.train_summary_writer.as_default():
        #     tf.summary.trace_export(
        #         name="veos_trace",
        #         step=epi_cnt_local,
        #         profiler_out_dir=self.train_log_dir,
        #     )
        self.agent.buffer.close()
        plt.close(fig="all")

        self.logger_control_flow.info(
            f"{{'header': 'Close Buffer, pool!'}}", extra=self.dictLogger
        )
        self.thr_observe.join()
        if self.cloud:
            self.thr_remote_get.join()
        self.thr_flash.join()
        self.thr_countdown.join()

        self.logger_control_flow.info(
            f"{{'header': 'main dies!!!!'}}", extra=self.dictLogger
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
        app = Avatar(
            truck=truck,
            driver=driver,
            can_server=can_server,
            trip_server=trip_server,
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
        main_run = app.infer
    else:
        main_run = app.train

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        executor.submit(main_run)
        executor.submit(app.pipeline.train)
        executor.submit(app.thr_countdown.join)
        if args.cloud:
            executor.submit(app.thr_remote_get.join)
        else:
            executor.submit(app.thr_remote_can.join)
