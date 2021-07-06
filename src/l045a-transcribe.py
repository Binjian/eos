import os
import time

import os
import time
import signal
import subprocess

fileName = "../data/udp-pcap/veos_L025.asc.pcap"  # file name
portNum = 8002  # port number
p = os.execlp(
    "tcpdump", "udp", "-w", fileName, "-i", "lo", "port", str(portNum)
)
# pid = os.fork()
# result = os.waitpid(-1, 0)
# if pid == 0:  # copy process
#     time.sleep(1)
#     # os.execlp("python", "python", "l045a_ac_tf.py")  #  run Simulation
# else:
#     p = subprocess.popen(
#         ["tcpdump", "udp", "-w", filename, "-i", "lo", "port", str(portnum)],
#         stdout=subprocess.pipe,
#     )
#     result = os.waitpid(-1, 0)
#     p.terminate()
