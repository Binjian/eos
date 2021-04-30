#!/usr/bin/env python3

import tbox_sim

if __name__ == "__main__":
    # args = parse_arg()
    value = [99.0] * 21 * 17
    tbox_sim.set_tbox_sim_path(
        "/home/is/devel/carla-drl/drl-carla-manual/src/comm/tbox"
    )
    tbox_sim.send_float_array("TQD_trqTrqSetECO_MAP_v", value)
