#!/usr/bin/env python3

import socket
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from scipy import interpolate

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
    for i in range(1, 36):
        send_str = send_str + " " + (str(table[i]))

    # s.sendto(send_str, ('192.168.8.107', 13251))
    # s.sendto(send_str, ('127.0.0.1', 13251))
    s.close()


def generate_vcu_calibration(
    npd, pedal_range, nvl, velocity_range
):  # input : npd 17, nvl 21; output vcu_param_list as float32
    pd = np.linspace(pedal_range[0], pedal_range[1], num=npd)  # 0 - 100% pedal
    vl = np.linspace(
        velocity_range[0], velocity_range[1], num=nvl
    )  # 0 - 72kmph velocity
    pdv, vlv = np.meshgrid(pd, vl, sparse=True)
    v = pdv / (1 + np.sqrt(np.abs(vlv)))

    return v
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # # Plot the surface.
    # surf = ax.plot_surface(pdv, vlv, v, cmap=cm.coolwarm,
    #                        linewidth=0, antialiased=False)
    # # Customize the z axis.
    # ax.set_zlim(-0.01, 1.01)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # # A StrMethodFormatter is used automatically
    # ax.zaxis.set_major_formatter('{x:.02f}')
    #
    # # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.show()
    # return np.float32(v)


def generate_lookup_table(
    pedal_range, velocity_range, calib_table
):  # input : npd 17, nvl 21; output vcu_param_list as float32
    nvl, npd = calib_table.shape
    p_step = complex(0, npd)
    v_step = complex(0, nvl)
    grid_v, grid_p = np.mgrid[
        velocity_range[0] : velocity_range[1] : v_step,
        pedal_range[0] : pedal_range[1] : p_step,
    ]
    calib_lookup = interpolate.interp2d(grid_p, grid_v, calib_table, kind="cubic")
    return calib_lookup


def prepare_vcu_calibration_table(table):
    vcu_param_list = table.astpye("float32")
    return vcu_param_list


#
# # env = Environment()
# send_table(test_list)
