import os
import time

import os
import time
import signal
import subprocess

fileName = "../data/udp-pcap/data.pcap"  # file name
portNum = 8002  # port number
pid = os.fork()
if pid == 0:  # copy process
    time.sleep(1)
    os.execlp("python", "python", "l045a_ac_tf.py")  #  run Simulation
else:
    p = subprocess.Popen(
        ["tcpdump", "udp", "-w", fileName, "-i", "lo", "port", str(portNum)],
        stdout=subprocess.PIPE,
    )
    result = os.waitpid(-1, 0)
    p.terminate()
