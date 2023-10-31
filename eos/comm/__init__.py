from pathlib import Path

from .remote.remote_can_client import (
    ClearablePullConsumer,  # type: ignore
    RemoteCanClient,
    RemoteCanException,
)
from .tbox.scripts.tbox_sim import send_float_array, set_tbox_sim_path  # type: ignore

from .tbox.utils.tbox_can_exceptions import TBoxCanException

set_tbox_sim_path(str(Path(__file__).parent) + "/tbox")
kvaser_send_float_array = send_float_array


__all__ = [
    "RemoteCanClient",
    "RemoteCanException",
    "kvaser_send_float_array",
    "TBoxCanException",
    "ClearablePullConsumer",
]
