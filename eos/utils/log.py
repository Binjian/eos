"""
logging service for the package
"""


import datetime
import inspect

# Logging Service Initialization
import logging

# system imports
import os
from logging.handlers import SocketHandler

from pythonjsonlogger import jsonlogger

# third-party imports


# logging.basicConfig(level=logging.DEBUG, format=fmt)
mpl_logger = logging.getLogger("matplotlib.font_manager")
mpl_logger.disabled = True


# logging.basicConfig(level=logging.DEBUG, format=fmt)
mpl_logger = logging.getLogger("matplotlib.font_manager")
mpl_logger.disabled = True
# logging.basicConfig(format=fmt)
logger = logging.getLogger("eos")
logger.propagate = False
dictLogger = {"user": inspect.currentframe().f_code.co_name}  # type: ignore


def get_logger(folder, name, level=logging.INFO):
    """get a logger for the given name
    args:
        folder (str): folder to store the log files
        name (str): name of the logger
        level (int): logging level （INFO，DEBUG，ERROR, ...）
    """
    # logging.basicConfig(format=fmt)
    mylogger = logging.getLogger(name)
    mylogger.setLevel(level)
    mylogger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s-%(name)s-%(levelname)s-%(module)s-%(threadName)s-%(funcName)s)-%(lineno)d): %(message)s"
    )
    json_file_formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(module)s %(threadName)s %(funcName)s) %(lineno)d) %(message)s"
    )

    logfolder = folder + "/py_logs/" + name
    try:
        os.makedirs(logfolder)
    except FileExistsError:
        print("User folder exists, just resume!")

    log_file_name = logfolder + (
        name + datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S") + ".log"
    )

    fh = logging.FileHandler(log_file_name)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(json_file_formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    #  Cutelog socket
    sh = SocketHandler("127.0.0.1", 19996)
    sh.setFormatter(formatter)

    mylogger.addHandler(fh)
    mylogger.addHandler(ch)
    mylogger.addHandler(sh)

    # dictLogger = {'funcName': '__self__.__func__.__name__'}
    # dictLogger = {'user': inspect.currentframe().f_back.f_code.co_name}
    mydictLogger = {"user": inspect.currentframe().f_code.co_name}

    return mylogger, mydictLogger
