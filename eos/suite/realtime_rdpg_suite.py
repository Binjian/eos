import os
import time

import os
import time
import datetime
import subprocess
import argparse

# resumption settings
parser = argparse.ArgumentParser(
    "DDPG with reduced observations (no expected velocity) Suite"
)
parser.add_argument(
    "-r",
    "--resume",
    help="resume the last training with restored model, checkpoint and pedal map",
    action="store_true",
)
parser.add_argument(
    "-t",
    "--record_table",
    help="record action table during training",
    action="store_true",
)
parser.add_argument(
    "-p",
    "--path",
    type=str,
    help="relative path to be saved, for create subfolder for different drivers",
)
args = parser.parse_args()

udpfileName = (
    os.getcwd()
    + "/../../data/udp-pcap/realtime_rdpg-"
    + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    + ".pcap"
)
portNum = 8002  # port number
pid = os.fork()
if pid == 0:  # copy process
    time.sleep(1)
    if args.resume:
        os.execlp(
            "python",
            "python",
            "../realtime_train_infer_rdpg.py",
            "--resume",
            "--path",
            args.path,
            "--record_table",
        )  #  run Simulation

    else:
        os.execlp(
            "python",
            "python",
            "../realtime_train_infer_rdpg.py",
            "--path",
            args.path,
            "--record_table",
        )  #  run Simulation
else:
    p = subprocess.Popen(
        ["tcpdump", "udp", "-w", udpfileName, "-i", "lo", "port", str(portNum)],
        stdout=subprocess.PIPE,
    )
    # output p status

    result = os.waitpid(-1, 0)
    p.terminate()
