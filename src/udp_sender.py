#!/usr/bin/env python3

import socket
import numpy as np

# test_list = range(1, 37)
# for i in range(36):
#   test_list[i] = test_list[i] + 0.01

def send_table(table):
  if len(table) != 36:
    print("table length must be 36")
    return
  # print(table)
  s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

  send_str = str(table[0])
  for i in range(1,36):
      send_str = send_str + ' ' + (str(table[i]))

  # s.sendto(send_str, ('192.168.8.107', 13251))
  #s.sendto(send_str, ('127.0.0.1', 13251))
  s.close()

def generate_vcu_calibration(k1, k2, kk):
  vcu_param_list = np.arange(36) + 0.01
  return vcu_param_list


#
# # env = Environment()
# send_table(test_list)
