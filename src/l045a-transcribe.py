import os

fileName = "../data/udp-pcap/veos_L025.asc.pcap"  # file name
portNum = 8002  # port number
p = os.execlp(
    "tcpdump", "udp", "-w", fileName, "-i", "lo", "port", str(portNum)
)
