import eos.comm.remotecan.remote_can_client.remote_can_client as remote_can_client
import eos.comm.tbox.scripts.tbox_sim as tbox_sim
import eos.comm.vcu_calib_generator as vcu_calib_generator
from pathlib import Path
import logging

projroot = Path(__file__).parent.parent

# TODO: Add logging support

# tracer = VizTracer()
# logging.basicConfig(level=logging.DEBUG, format=fmt)
mpl_logger = logging.getLogger("matplotlib.font_manager")
mpl_logger.disabled = True

# logging.basicConfig(format=fmt)
logger = logging.getLogger("__name__")
logger.propagate = False

__all__ = [remote_can_client, tbox_sim, vcu_calib_generator, projroot, logger]
