import os
import time

fileName = '../data/udp-pcap/data.pcap' # file name
portNum = 8002 # port number
sudoPassword = 'asdf' # system password
command = 'tcpdump udp -w '+ fileName +' -i lo port ' + str(portNum) # system command

pid = os.fork()
if pid == 0:  # copy process
    time.sleep(1)
    os.system('echo %s|sudo -S %s' % (sudoPassword, command))  # catch packets via tcpdump
    assert False, "error starting program"  # shouldn't return
else:
    print("Child is", pid)    
    os.execlp("python", "python", "l045a_ac_tf.py")  #  run Simulation