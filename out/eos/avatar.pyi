import abc
import logging
import queue
import threading
from collections import deque
from pathlib import Path
from threading import Lock, Thread
from typing import Optional, Union

import pandas as pd
import rocketmq.client as rmq_client
from _typeshed import Incomplete
from tensorflow.summary import SummaryWriter as SummaryWriter
from tensorflow.summary import scalar as scalar

from eos import projroot as projroot
from eos.agent import DDPG as DDPG
from eos.agent import DPG as DPG
from eos.agent import RDPG as RDPG
from eos.agent.utils import hyper_param_by_name as hyper_param_by_name
from eos.comm import ClearablePullConsumer as ClearablePullConsumer
from eos.comm import RemoteCanClient as RemoteCanClient
from eos.comm import RemoteCanException as RemoteCanException
from eos.comm import kvaser_send_float_array as kvaser_send_float_array
from eos.data_io.config import CANMessenger as CANMessenger
from eos.data_io.config import Driver as Driver
from eos.data_io.config import TripMessenger as TripMessenger
from eos.data_io.config import Truck as Truck
from eos.data_io.config import TruckInCloud as TruckInCloud
from eos.data_io.config import TruckInField as TruckInField
from eos.data_io.config import str_to_can_server as str_to_can_server
from eos.data_io.config import str_to_driver as str_to_driver
from eos.data_io.config import str_to_trip_server as str_to_trip_server
from eos.data_io.config import str_to_truck as str_to_truck
from eos.utils import GracefulKiller as GracefulKiller
from eos.utils import assemble_action_ser as assemble_action_ser
from eos.utils import assemble_reward_ser as assemble_reward_ser
from eos.utils import assemble_state_ser as assemble_state_ser
from eos.utils import dictLogger as dictLogger
from eos.utils import logger as logger
from eos.utils import ragged_nparray_list_interp as ragged_nparray_list_interp
from eos.visualization import plot_3d_figure as plot_3d_figure
from eos.visualization import plot_to_image as plot_to_image

class Avatar(abc.ABC):
    truck: Truck
    driver: Driver
    can_server: CANMessenger
    trip_server: TripMessenger
    cloud: bool
    ui: str
    logger: logging.Logger
    resume: bool
    infer_mode: bool
    record: bool
    path: str
    pool_key: str
    proj_root: Path
    data_root: Path
    table_root: Path
    program_start: bool
    program_exit: bool
    vel_hist_dq: Optional[deque]
    train_summary_writer: Optional[SummaryWriter]
    remotecan_client: Optional[RemoteCanClient]
    rmq_consumer: Optional[ClearablePullConsumer]
    rmq_message_ready: rmq_client.Message
    rmq_producer: Optional[rmq_client.Producer]
    log_root: Path
    logger_control_flow: Optional[logging.Logger]
    tflog: Optional[logging.Logger]
    vcu_calib_table0: Optional[pd.DataFrame]
    vcu_calib_table1: Optional[pd.DataFrame]
    hmi_lock: Optional[Lock]
    state_machine_lock: Optional[Lock]
    tableQ_lock: Optional[Lock]
    captureQ_lock: Optional[Lock]
    remoteClient_lock: Optional[Lock]
    flash_env_lock: Optional[Lock]
    get_env_lock: Optional[Lock]
    done_env_lock: Optional[Lock]
    tableQueue: Optional[queue.Queue]
    motion_power_queue: Optional[queue.Queue]
    episode_done: bool
    episode_end: bool
    episode_count: int
    step_count: int
    epi_countdown_time: float
    get_truck_status_start: bool
    epi_countdown: bool
    get_truck_status_motion_power_t: list
    get_truck_status_myHost: str
    get_truck_status_myPort: int
    get_truck_status_qobject_len: int
    vcu_calib_table_row_start: int
    thr_countdown: Optional[Thread]
    thr_observe: Optional[Thread]
    thr_remote_get: Optional[Thread]
    thr_flash: Optional[Thread]
    repo: Incomplete
    dictLogger: Incomplete
    eps: Incomplete
    get_truck_status: Incomplete
    flash_vcu: Incomplete
    def __post_init__(self) -> None: ...
    @property
    def agent(self) -> Union[DPG | None]: ...
    def init_cloud(self) -> None: ...
    def set_logger(self) -> None: ...
    def set_data_path(self) -> None: ...
    def init_vehicle(self) -> None: ...
    def init_threads_data(self) -> None: ...
    def capture_countdown_handler(
        self,
        evt_epi_done: threading.Event,
        evt_remote_get: threading.Event,
        evt_remote_flash: threading.Event,
    ): ...
    def kvaser_get_truck_status(
        self,
        evt_epi_done: threading.Event,
        evt_remote_get: threading.Event,
        evt_remote_flash: threading.Event,
    ): ...
    def kvaser_flash_vcu(self, evt_remote_flash: threading.Event): ...
    def remote_get_handler(
        self, evt_remote_get: threading.Event, evt_remote_flash: threading.Event
    ): ...
    def remote_hmi_rmq_state_machine(
        self,
        evt_epi_done: threading.Event,
        evt_remote_get: threading.Event,
        evt_remote_flash: threading.Event,
    ): ...
    def remote_hmi_no_state_machine(
        self,
        evt_epi_done: threading.Event,
        evt_remote_get: threading.Event,
        evt_remote_flash: threading.Event,
    ): ...
    def remote_hmi_tcp_state_machine(
        self,
        evt_epi_done: threading.Event,
        evt_remote_get: threading.Event,
        evt_remote_flash: threading.Event,
    ): ...
    def remote_flash_vcu(self, evt_remote_flash: threading.Event): ...
    def run(self) -> None: ...
    def __init__(
        self,
        *,
        truck,
        driver,
        can_server,
        trip_server,
        _agent,
        cloud,
        ui,
        logger,
        resume,
        infer_mode,
        record,
        path,
        pool_key,
        proj_root,
        data_root,
        table_root,
        program_start,
        program_exit,
        vel_hist_dq,
        train_summary_writer,
        remotecan_client,
        rmq_consumer,
        rmq_message_ready,
        rmq_producer,
        log_root,
        logger_control_flow,
        tflog,
        vcu_calib_table0,
        vcu_calib_table1,
        hmi_lock,
        state_machine_lock,
        tableQ_lock,
        captureQ_lock,
        remoteClient_lock,
        flash_env_lock,
        get_env_lock,
        done_env_lock,
        tableQueue,
        motion_power_queue,
        episode_done,
        episode_end,
        episode_count,
        step_count,
        epi_countdown_time,
        get_truck_status_start,
        epi_countdown,
        get_truck_status_motion_power_t,
        get_truck_status_myHost,
        get_truck_status_myPort,
        get_truck_status_qobject_len,
        vcu_calib_table_row_start,
        thr_countdown,
        thr_observe,
        thr_remote_get,
        thr_flash
    ) -> None: ...
