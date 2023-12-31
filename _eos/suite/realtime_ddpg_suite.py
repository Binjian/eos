import argparse
import datetime
import os
import subprocess
import time

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
    + "/../../data/udp-pcap/realtime_ddpg_vb-"
    + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    + ".pcap"
)
# --cloud -t -p testremote -r
portNum = 8002  # port number
pid = os.fork()
if pid == 0:  # copy process
    time.sleep(1)
    if args.resume:
        os.execlp(
            "python",
            "python",
            "../realtime_train_infer_ddpg.py",
            "--resume",
            "--cloud",
            "--web",
            "--path",
            args.path,
            "--record_table",
        )  #  run Simulation

    else:
        os.execlp(
            "python",
            "python",
            "../realtime_train_infer_ddpg.py",
            "--cloud",
            "--web",
            "--path",
            args.path,
            "--record_table",
        )  #  run Simulation
else:
    p = subprocess.Popen(
        [
            "tcpdump",
            "udp",
            "-w",
            udpfileName,
            "-i",
            "lo",
            "port",
            str(portNum),
        ],
        stdout=subprocess.PIPE,
    )
    result = os.waitpid(-1, 0)
    p.terminate()
