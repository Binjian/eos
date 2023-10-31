from _typeshed import Incomplete

from eos import projroot as projroot
from eos.utils import dictLogger as dictLogger
from eos.utils import logger as logger

from .agent.rdpg import RDPG as RDPG
from .agent.utils import HYPER_PARAM as HYPER_PARAM
from .agent.utils import hyper_param_by_name as hyper_param_by_name
from .avatar import Avatar as Avatar

class AvatarRDPG(Avatar):
    hyper_param: HYPER_PARAM
    agent: Incomplete
    def __post_init__(self) -> None: ...
    def __init__(
        self,
        hyper_param,
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
