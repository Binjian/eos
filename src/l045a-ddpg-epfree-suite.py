import os
import time

import os
import time
import datetime
import subprocess
import argparse

# resumption settings
parser = argparse.ArgumentParser("DDPG episode free without training Suite")
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
    "-i",
    "--infer",
    help="No model update and training. Only Inference",
    action="store_false",
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
    + "/../data/udp-pcap/l045a_ddpg_epfree-"
    + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    + ".pcap"
)

portNum = 8002  # port number
pid = os.fork()
if pid == 0:  # copy process
    time.sleep(1)
    if args.resume:
        if not args.infer:
            os.execlp(
                "python",
                "python",
                "l045a_ddpg_epfree.py",
                "--resume",
                "--path",
                args.path,
                "--record_table",
            )
        else:
            os.execlp(
                "python",
                "python",
                "l045a_ddpg_epfree.py",
                "--resume",
                "--infer",
                "--path",
                args.path,
                "--record_table",
            )
    else:
        if not args.infer:
            os.execlp(
                "python",
                "python",
                "l045a_ddpg_epfree.py",
                "--path",
                args.path,
                "--record_table",
            )  #  run Simulation
        else:
            os.execlp(
                "python",
                "python",
                "l045a_ddpg_epfree.py",
                "--infer",
                "--path",
                args.path,
                "--record_table",
            )  # run Simulation
else:
    p = subprocess.Popen(
        ["tcpdump", "udp", "-w", udpfileName, "-i", "lo", "port", str(portNum)],
        stdout=subprocess.PIPE,
    )
    result = os.waitpid(-1, 0)
    p.terminate()
