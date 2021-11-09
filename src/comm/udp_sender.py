#!/usr/bin/env python3

import socket
import numpy as np
import json


def send_table(table):
    if len(table) != 36:
        print("table length must be 36")
        return
    # print(table)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    send_str = str(table[0])
    for i in range(1, 36):
        send_str = send_str + " " + (str(table[i]))

    # s.sendto(send_str, ('192.168.8.107', 13251))
    # s.sendto(send_str, ('127.0.0.1', 13251))
    s.close()


def prepare_vcu_calibration_table(table):
    vcu_param_list = table.astpye("float32")
    return vcu_param_list


def get_table_from_json(json_file):
    with open(json_file, "r") as f:
        table = json.load(f)
    return table
