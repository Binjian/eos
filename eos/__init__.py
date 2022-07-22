from .comm.remote.remote_can_client.remote_can_client import RemoteCan
from .comm.remote.remote_can_client.offline_experiences import RecordPool
from .comm.tbox.scripts import tbox_sim
from .comm.vcu_calib_generator import generate_vcu_calibration
from pathlib import Path
import logging
import inspect

projroot = Path(__file__).parent.parent

# TODO: Add logging support

# tracer = VizTracer()
# logging.basicConfig(level=logging.DEBUG, format=fmt)
mpl_logger = logging.getLogger("matplotlib.font_manager")
mpl_logger.disabled = True

# logging.basicConfig(format=fmt)
logger = logging.getLogger("eos")
logger.propagate = False
dictLogger = {"user": inspect.currentframe().f_code.co_name}
__all__ = [
    RemoteCan,
    tbox_sim,
    generate_vcu_calibration,
    projroot,
    logger,
    dictLogger,
]
