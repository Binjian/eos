import os
import datetime
from pythonjsonlogger import jsonlogger

# Logging Service Initialization
import logging
from logging.handlers import SocketHandler
import inspect
import numpy as np

# internal import
from eos.comm import generate_vcu_calibration, send_float_array, set_tbox_sim_path


mpl_logger = logging.getLogger("matplotlib.font_manager")
mpl_logger.disabled = True

# logging.basicConfig(format=fmt)
logger = logging.getLogger("l045a")
logger.propagate = False
formatter = logging.Formatter(
    "%(asctime)s-%(name)s-%(levelname)s-%(module)s-%(threadName)s-%(funcName)s)-%(lineno)d): %(message)s"
)
json_file_formatter = jsonlogger.JsonFormatter(
    "%(asctime)s %(name)s %(levelname)s %(module)s %(threadName)s %(funcName)s) %(lineno)d) %(message)s"
)

datafolder = "../../data"
logfolder = datafolder + "/py_logs"
try:
    os.makedirs(logfolder)
except FileExistsError:
    print("User folder exists, just resume!")

logfilename = logfolder + (
    "/l045a-transcribe-"
    + datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    + ".log"
)

fh = logging.FileHandler(logfilename)
fh.setLevel(logging.DEBUG)
fh.setFormatter(json_file_formatter)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
sh = SocketHandler("127.0.0.1", 19996)
sh.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)
logger.addHandler(sh)

logger.setLevel(logging.DEBUG)
# dictLogger = {'funcName': '__self__.__func__.__name__'}
# dictLogger = {'user': inspect.currentframe().f_back.f_code.co_name}
dictLogger = {"user": inspect.currentframe().f_code.co_name}

logc = logger.getChild("control flow")
logc.propagate = True
logd = logger.getChild("data flow")
logd.propagate = True

vcu_calib_table_col = 17  # number of pedal steps, x direction
vcu_calib_table_row = 21  # numnber of velocity steps, y direction

pedal_range = [0, 1.0]
velocity_range = [0, 20.0]


vcu_calib_table0 = generate_vcu_calibration(
    vcu_calib_table_col,
    pedal_range,
    vcu_calib_table_row,
    velocity_range,
    2,
    datafolder,
)

set_tbox_sim_path(os.getcwd() + "/comm/tbox")

vcu_calib_table1 = np.copy(vcu_calib_table0)  # shallow copy of the default table
vcu_table1 = vcu_calib_table1.reshape(-1).tolist()
logger.info(f"Start flash initial table", extra=dictLogger)
# time.sleep(1.0)
send_float_array("TQD_trqTrqSetNormal_MAP_v", vcu_table1, sw_diff=False)
logger.info(f"Done flash initial table", extra=dictLogger)
# TQD_trqTrqSetECO_MAP_v

udp_logfilename = (
    datafolder
    + "/udp-pcap/l045a-noAI-"
    + datetime.datetime.now().strftime("%y-%m-%d-%h-%m-%s_%f")[:-3]
    + ".pcap"
)
portNum = 8002  # port number
p = os.execlp("tcpdump", "udp", "-w", udp_logfilename, "-i", "lo", "port", str(portNum))
