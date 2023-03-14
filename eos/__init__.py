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

from .comm import Pool, MongoStore, NPAStore, RemoteCan, ClearablePullConsumer
from .comm import kvaser_send_float_array
from .config.vcu_calib_generator import generate_vcu_calibration
from .realtime_train_infer_ddpg import RealtimeDDPG
from .realtime_train_infer_rdpg import RealtimeRDPG
from .algo import DPG

projroot = Path(__file__).parent.parent

# TODO: Add logging support

# tracer = VizTracer()


__all__ = [
    RemoteCan,
    RealtimeDDPG,
    RealtimeRDPG,
    Pool,
    MongoStore,
    NPAStore,
    RemoteCan,
    ClearablePullConsumer,
    kvaser_send_float_array,
    generate_vcu_calibration,
    projroot,
    logger,
    dictLogger,
]
