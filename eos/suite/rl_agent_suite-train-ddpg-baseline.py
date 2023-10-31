import datetime
import os
import subprocess
import time

udpfileName = (
    os.getcwd()
    + "/../../data/udp-pcap/rl_agent_vb-ddpg-baseline-"
    + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    + ".pcap"
)
# --cloud -t -p testremote -r
portNum = 8002  # port number
pid = os.fork()
if pid == 0:  # copy process
    time.sleep(1)
    os.execlp(
        "python",
        "python",
        "../rl_agent.py",
        "--agent",
        "ddpg",
        "--ui",
        "local",
        "--resume",
        "--path",
        "baseline",
        "--record_table",
        "-v",
        "VB7",
        "-d",
        "longfei",
        "-o",
        "mongo_local",
    )

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
