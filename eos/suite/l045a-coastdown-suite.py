import argparse
import datetime
import os
import subprocess
import time

# resumption settings
parser = argparse.ArgumentParser()
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
    + "/../../data/udp-pcap/l045a_ac_tf_coastdown-noaircond-"
    + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S_%f")[:-3]
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
            "../l045a_ac_tf_coastdown.py",
            "--resume",
            "--path",
            args.path,
            "--record_table",
        )  #  run Simulation
    else:
        os.execlp("python", "python", "../l045a_ac_tf_coastdown.py")  #  run Simulation
else:
    p = subprocess.Popen(
        ["tcpdump", "udp", "-w", udpfileName, "-i", "lo", "port", str(portNum)],
        stdout=subprocess.PIPE,
    )
    result = os.waitpid(-1, 0)
    p.terminate()
