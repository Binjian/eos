import inspect
import logging
from pathlib import Path

# logging.basicConfig(level=logging.DEBUG, format=fmt)
mpl_logger = logging.getLogger("matplotlib.font_manager")
mpl_logger.disabled = True
# logging.basicConfig(format=fmt)
logger = logging.getLogger("eos")
logger.propagate = False
dictLogger = {"user": inspect.currentframe().f_code.co_name}

from .comm.remote.remote_can_client.pool import Pool
from .comm.remote.remote_can_client.remote_can_client import RemoteCan
from .comm.tbox.scripts import tbox_sim
from .config.vcu_calib_generator import generate_vcu_calibration

projroot = Path(__file__).parent.parent

# TODO: Add logging support

# tracer = VizTracer()


__all__ = [
    RemoteCan,
    Pool,
    tbox_sim,
    generate_vcu_calibration,
    projroot,
    logger,
    dictLogger,
]
