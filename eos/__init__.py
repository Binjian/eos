import inspect
import logging
from pathlib import Path

# logging.basicConfig(level=logging.DEBUG, format=fmt)
mpl_logger = logging.getLogger('matplotlib.font_manager')
mpl_logger.disabled = True
# logging.basicConfig(format=fmt)
logger = logging.getLogger('eos')
logger.propagate = False
dictLogger = {'user': inspect.currentframe().f_code.co_name}

from .comm import (
    Pool,
    DBPool,
    RecordFilePool,
    EpisodeFilePool,
    RemoteCan,
    ClearablePullConsumer,
)
from .comm import kvaser_send_float_array
from .config.vcu_calib_generator import generate_vcu_calibration
from .algo import DPG
from .algo import DDPG
from .algo import RDPG

projroot = Path(__file__).parent.parent

# TODO: Add logging support

# tracer = VizTracer()


__all__ = [
    DPG,
    DDPG,
    RDPG,
    RemoteCan,
    Pool,
    DBPool,
    RecordFilePool,
    EpisodeFilePool,
    RemoteCan,
    ClearablePullConsumer,
    kvaser_send_float_array,
    generate_vcu_calibration,
    projroot,
    logger,
    dictLogger,
]
