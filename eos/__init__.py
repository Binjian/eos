from .comm.remotecan.remote_can_client import remote_can_client
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
logger = logging.getLogger("__name__")
logger.propagate = False
dictLogger = {"user": inspect.currentframe().f_code.co_name}
__all__ = [
    remote_can_client,
    tbox_sim,
    generate_vcu_calibration,
    projroot,
    logger,
    dictLogger,
]
