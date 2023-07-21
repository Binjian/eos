from pathlib import Path

from .remote.remote_can_client import (
    ClearablePullConsumer,
    RemoteCan,
)
from .tbox.scripts.tbox_sim import send_float_array, set_tbox_sim_path

set_tbox_sim_path(str(Path(__file__).parent) + '/tbox')
kvaser_send_float_array = send_float_array


__all__ = [
    'RemoteCan',
    'kvaser_send_float_array',
    'ClearablePullConsumer',
]
