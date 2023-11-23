import datetime
import inspect

# Logging Service Initialization
import logging
import os
from logging.handlers import SocketHandler
from pathlib import Path


# internal import
from eos.data_io.config.vcu_calib_generator import generate_vcu_calibration
from eos.comm import kvaser_send_float_array
from eos.data_io.utils.log import set_root_logger
from eos.data_io.config.vehicles import trucks_by_id, Truck
from eos.data_io.config.drivers import drivers_by_id
from eos.comm import TBoxCanException

# mpl_logger = logging.getLogger("matplotlib.font_manager")
# mpl_logger.disabled = True

# logging.basicConfig(format=fmt)
datafolder = Path("../../data")
truck = trucks_by_id["VB7"]
driver = drivers_by_id["default"]
logger, dict_logger = set_root_logger(
    "eos",
    data_root=datafolder,
    agent="agent",
    tz=truck.tz,
    truck=truck.vid,
    driver=driver.pid,
)

vcu_calib_table_col = 17  # number of pedal steps, x direction
vcu_calib_table_row = 14  # numnber of velocity steps, y direction

pedal_range = [0, 1.0]
velocity_range = [0, 120.0]


vcu_calib_table0 = generate_vcu_calibration(
    vcu_calib_table_col,
    pedal_range,
    vcu_calib_table_row,
    velocity_range,
    2,
    Path('../data_io/config/')
)


logger.info(f"Start flash initial table", extra=dict_logger)
try:
    kvaser_send_float_array(vcu_calib_table0, sw_diff=False)
except TBoxCanException as exc:
    raise exc
except Exception as exc:
    raise exc

logger.info(
    f"{{'header': 'Done with flashing table'}}",
    extra=dict_logger,
)

udp_logfilename = (
    str(datafolder)
    + "/udp-pcap/l045a-noAI-"
    + datetime.datetime.now().strftime("%y-%m-%d-%h-%m-%s_%f")[:-3]
    + ".pcap"
)
portNum = 8002  # port number
p = os.execlp("tcpdump", "udp", "-w", udp_logfilename, "-i", "lo", "port", str(portNum))  # type: ignore
