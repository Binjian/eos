from .remotecan.remote_can_client.remote_can_client import RemoteCan
from .tbox.scripts.tbox_sim import set_tbox_sim_path, send_float_array
from .vcu_calib_generator import generate_vcu_calibration, generate_lookup_table
from pathlib import Path

set_tbox_sim_path(str(Path(__file__).parent) + "/tbox")
kvaser_send_float_array = send_float_array


__all__ = [
    "RemoteCan",
    "kvaser_send_float_array",
    "generate_vcu_calibration",
    "generate_lookup_table",
]
