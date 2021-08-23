import os
import datetime

udp_logfilename = (
    "../data/udp-pcap/l045a-noAI-"
    + datetime.datetime.now().strftime("%y-%m-%d-%h-%m-%s_%f")[:-3]
    + ".pcap"
)
portNum = 8002  # port number
p = os.execlp("tcpdump", "udp", "-w", udp_logfilename, "-i", "lo", "port", str(portNum))
