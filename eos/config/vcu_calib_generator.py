#!/usr/bin/env python3

import glob
import os
import os.path
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import interpolate


# for real vcu, values in the table will be the requrested torque
# Current throttlel (0,1) should be a coefficient of multplicative factor
# like between +/- 20% or empirically give safety bounds.
# action space will be then within this bounds
# TODO ask for safety bounds and real vcu to be integrated.
# TODO generate a mask according to WLTC to reduce parameter optimization space.


def generate_vcu_calibration(  # pedal is x(column), velocity is y(row) )
    npd, pedal_range, nvl, velocity_range, shortcut, dataroot
):  # input : npd 17, nvl 21; output vcu_param_list as float32
    """
    Generate VCU calibration parameters for a given truck.
    return: pandas dataframe
    """
    ped = np.linspace(pedal_range[0], pedal_range[1], num=npd)  # 0 - 100% pedal

    if shortcut == 1:
        vel_ = np.linspace(
            velocity_range[0], velocity_range[1], num=nvl - 1
        )  # 0 - 120 kmph velocity
        vel = np.insert(vel_, 1, 7)  # insert 7 kmph, and convert to m/s
        pdv, vlv = np.meshgrid(ped, vel / 3.6, sparse=True)
        v = pdv / (1 + np.sqrt(np.abs(vlv)))
        pd_v = pd.DataFrame(v, columns=ped, index=vel)

    elif shortcut == 2:  # import default eco calibration table
        table_path = dataroot.joinpath(
            "vb7_init_table.csv"
        )  # init table is driver independent in the pardir.
        pd_v = pd.read_csv(table_path, header=0, index_col=0)
        # table_path = datafolder + "/54_vertices_approx-regen3.csv"  # init table is driver independent in the pardir.
        # pd_data = pd.read_csv(table_path, header=0, index_col=0)
    elif shortcut == 3:  # import latest pedal map that was used
        files = glob.glob(str(dataroot) + "/last_table*.csv")
        if not files:  # files is empty list []
            print("no last table is available. Get init table instead.")
            latest_table = Path(__file__).parent.joinpath(
                "vb7_init_table.csv"
            )  # init table is driver independent in the pardir.
        else:
            latest_table = max(files, key=os.path.getmtime)
        # latest = datafolder + "last_table.csv"  # init table is driver relevant.
        pd_v = pd.read_csv(latest_table, header=0, index_col=0)

    else:
        vel = np.ones(nvl)
        pdv, vlv = np.meshgrid(ped, vel, sparse=False)
        pd_v = pd.DataFrame(pdv, columns=ped, index=vel)
    return pd_v


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


def generate_lookup_table(  # pedal in x(col), velocity in y(row)
    pedal_range, velocity_range, calib_table
):  # input : npd 17, nvl 21; output vcu_param_list as float32
    nvl, npd = calib_table.shape
    p_step = complex(0, npd)
    v_step = complex(0, nvl)
    grid_v, grid_p = np.mgrid[
        velocity_range[0] : velocity_range[1] : v_step,
        pedal_range[0] : pedal_range[1] : p_step,
    ]
    calib_lookup = interpolate.interp2d(
        grid_p, grid_v, calib_table.to_numpy(), kind="linear"
    )
    return calib_lookup


if __name__ == "__main__":

    def test_generate_lookup_table():
        vcu_calib_table_row = 17  # number of pedal steps
        vcu_calib_table_col = 21  # numnber of velocity steps
        pedal_range = [0, 1.0]
        velocity_range = [0, 20.0]
        vcu_calib_table = generate_vcu_calibration(
            vcu_calib_table_row, pedal_range, vcu_calib_table_col, velocity_range
        )
        vcu_lookup_table = generate_lookup_table(
            pedal_range, velocity_range, vcu_calib_table
        )
        return vcu_lookup_table

    def test_generate_vcu_calibration():
        vcu_calib_table_row = 17  # number of pedal steps
        vcu_calib_table_col = 21  # numnber of velocity steps
        pedal_range = [0, 1.0]
        velocity_range = [0, 20.0]
        vcu_calib_table = generate_vcu_calibration(
            vcu_calib_table_row, pedal_range, vcu_calib_table_col, velocity_range
        )
        return vcu_calib_table

    vcu_calib_table = test_generate_vcu_calibration()
    vcu_lookup_table = test_generate_lookup_table()

    vcu_calib_table_row = 17  # number of pedal steps
    vcu_calib_table_col = 21  # numnber of velocity steps
    pedal_range = [0, 1.0]
    velocity_range = [0, 20.0]
    vcu_calib_table = generate_vcu_calibration(
        vcu_calib_table_row, pedal_range, vcu_calib_table_col, velocity_range
    )
    vcu_lookup_table = generate_lookup_table(
        pedal_range, velocity_range, vcu_calib_table
    )

    throt = 0
    speed = 0
    throttle = vcu_lookup_table(
        throt, speed
    )  # look up vcu table with pedal and speed  for throttle request
    print("throttle={}".format(throttle))

    throt = 1
    speed = 0
    throttle = vcu_lookup_table(
        throt, speed
    )  # look up vcu table with pedal and speed  for throttle request
    print("throttle={}".format(throttle))

    throt = 1
    speed = 10
    throttle = vcu_lookup_table(
        throt, speed
    )  # look up vcu table with pedal and speed  for throttle request
    print("throttle={}".format(throttle))
