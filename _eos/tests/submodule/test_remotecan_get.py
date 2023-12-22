# system import
# 3rd party import
import datetime
import inspect
import logging
import os
import unittest
import warnings

import numpy as np
import pandas as pd

from eos import proj_root

# local import
# import src.comm.remotecan.remote_can_client.remote_can_client as remote_can_client
from eos.comm.remote.remote_can_client import RemoteCanClient
from eos.data_io.config import (
    can_servers_by_name,
    generate_vcu_calibration,
    trip_servers_by_name,
    trucks_by_id,
    trucks_by_vin,
)
from eos.data_io.utils import ragged_nparray_list_interp

# import ...src.comm.remotecan.remote_can_client.remote_can_client

# ignore DeprecationWarning
warnings.filterwarnings("ignore", message="currentThread", category=DeprecationWarning)
# warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestRemoteCanGet(unittest.TestCase):
    """Tests for 'remote_can_client.py'."""

    site = "internal"

    def setUp(self) -> None:
        """Set up proxy and client"""
        self.proxies = {
            "http": "http://127.0.0.1:20171",
            "https": "http://127.0.0.1:20171",
        }
        self.proxies_socks = {
            "http": "socks5://127.0.0.1:20170",
            "https": "socks5://127.0.0.1:20170",
        }
        self.proxies_lantern = {
            "http": "http://127.0.0.1:34663",
            "https": "http://127.0.0.1:34663",
        }
        os.environ["http_proxy"] = ""  # for native test (internal site force no proxy)
        self.truck_name = "VB7"  # index of truck to test, 0 is VB7, 1 is VB6, 2 is HQ

        self.proj_root = proj_root
        self.logger = logging.getLogger("eostest")
        self.logger.propagate = False
        self.dict_logger = {"user": inspect.currentframe().f_code.co_name}
        self.truck = trucks_by_id[self.truck_name]
        self.set_logger(proj_root)

        # check if the truck is valid
        self.assertEqual(self.truck_name, self.truck.vid)

        latest_file = proj_root / "eos/data_io/config" / "vb7_init_table.csv"
        self.vcu_calib_table_default = pd.read_csv(latest_file, index_col=0)

    def set_logger(self, proj_root):
        log_root = proj_root.joinpath("data/scratch/tests")
        try:
            os.makedirs(log_root)
        except FileExistsError:
            pass
        logfile = log_root.joinpath(
            "test_remotecan_get-"
            + self.truck.vid
            + datetime.datetime.now().isoformat().replace(":", "-")
            + ".log"
        )

        formatter = logging.Formatter(
            "%(asctime)s-%(name)s-%(levelname)s-%(module)s-%(threadName)s-%(funcName)s)-%(lineno)d): %(message)s"
        )
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        self.logger.setLevel(logging.DEBUG)

    #
    # @unittest.skipIf(site == "internal", "skip for internal test")
    # def test_proxy_get(self):
    #
    #     self.logger.info("start test_proxy", extra=self.dict_logger)
    #     self.client = RemoteCanClient(
    #         vin=self.trucks[self.truck_ind].VIN, proxies=self.proxies_lantern
    #     )
    #     self.native_get()
    #
    # @unittest.skipIf(site == "internal", "skip for internal test")
    # def test_proxy_send(self):
    #
    #     self.logger.info("start test_proxy", extra=self.dict_logger)
    #     self.client = RemoteCanClient(
    #         vin=self.trucks[self.truck_ind].VIN, proxies=self.proxies_lantern
    #     )
    #     self.native_send()

    # @unittest.skipIf(site == "internal", "skip for internal test")
    # def test_native_get(self):
    #     self.logger.info("Start test_native_get", extra=self.dict_logger)
    #     self.client = RemoteCanClient(
    #         truck_name=self.truck.vid, url=self.truck.RemoteCANHost
    #     )
    #     self.logger.info("Set client", extra=self.dict_logger)
    #     self.native_get()

    # @unittest.skipIf(site == "internal", "skip for internal test")
    # def test_native_send(self):
    #     self.logger.info("Start test_native_send", extra=self.dict_logger)
    #     self.client = RemoteCanClient(
    #         truck_name=self.truck.vid, url=self.truck.RemoteCANHost
    #     )
    #     self.logger.info("Set client", extra=self.dict_logger)
    #
    #     self.native_send()

    def native_get(self):
        signal_success, remotecan_data = self.client.get_signals(
            duration=self.truck.CloudUnitNumber
        )
        self.logger.info(
            f"get_signal(), return state:{signal_success}",
            extra=self.dict_logger,
        )

        data_type = type(remotecan_data)
        self.logger.info(f"data type: {data_type}")
        if not isinstance(remotecan_data, dict):
            raise TypeError("udp sending wrong data type!")
        if signal_success == 0:
            try:
                # json_string = json.dumps(
                #     json_ret, indent=4, sort_keys=True, separators=(",", ": ")
                # )
                # print(f"print whole json string:{json_string}")

                self.logger.info("show remotecan_data", extra=self.dict_logger)
                signal_freq = self.truck.CloudSignalFrequency
                gear_freq = self.truck.CloudGearFrequency
                unit_duration = self.truck.CloudUnitDuration
                unit_ob_num = unit_duration * signal_freq
                unit_gear_num = unit_duration * gear_freq
                unit_num = self.truck.CloudUnitNumber
                timestamp_upsample_rate = (
                    self.truck.CloudSignalFrequency * self.truck.CloudUnitDuration
                )
                # timestamp_num = int(self.observe_length // duration)

                for key, value in remotecan_data.items():
                    if key == "result":
                        self.logger.info("show result", extra=self.dict_logger)

                        # timestamp processing
                        timestamps = []
                        separators = (
                            "--T::."  # adaption separators of the raw intest string
                        )
                        start_century = "20"
                        timezone = "+0800"
                        for ts in value["timestamps"]:
                            # create standard iso string datetime format
                            ts_substrings = [
                                ts[i : i + 2] for i in range(0, len(ts), 2)
                            ]
                            ts_iso = start_century
                            for i, sep in enumerate(separators):
                                ts_iso = ts_iso + ts_substrings[i] + sep
                            ts_iso = ts_iso + ts_substrings[-1] + timezone
                            timestamps.append(ts_iso)
                        timestamps_units = (
                            np.array(timestamps)
                            .astype("datetime64[ms]")
                            .astype("int")  # convert to int
                        )
                        if len(timestamps_units) != unit_num:
                            raise ValueError(
                                f"timestamps_units length is {len(timestamps_units)}, not {unit_num}"
                            )
                        # upsample gears from 2Hz to 50Hz
                        timestamps_seconds = list(timestamps_units)  # in ms
                        sampling_interval = 1.0 / signal_freq * 1000  # in ms
                        timestamps = [
                            i + j * sampling_interval
                            for i in timestamps_seconds
                            for j in np.arange(unit_ob_num)
                        ]
                        timestamps = np.array(timestamps).reshape(
                            (self.truck.CloudUnitNumber, -1)
                        )
                        self.logger.info(f"Timestamps{timestamps.shape}:{timestamps}")

                        # current = np.array(value["list_current_1s"])
                        current = ragged_nparray_list_interp(
                            value["list_current_1s"], ob_num=unit_ob_num
                        )
                        self.logger.info(f"current{current.shape}:{current}")

                        # voltage
                        voltage = ragged_nparray_list_interp(
                            value["list_voltage_1s"], ob_num=unit_ob_num
                        )
                        r_v, c_v = voltage.shape
                        # voltage needs to be upsampled in columns if its sample rate is half of the current
                        if c_v == current.shape[1] // 2:
                            voltage = np.repeat(voltage, 2, axis=1)
                        self.logger.info(f"voltage{voltage.shape}:{voltage}")

                        thrust = ragged_nparray_list_interp(
                            value["list_pedal_1s"], ob_num=unit_ob_num
                        )
                        self.logger.info(f"accl{thrust.shape}:{thrust}")

                        brake = ragged_nparray_list_interp(
                            value["list_brake_pressure_1s"], ob_num=unit_ob_num
                        )
                        self.logger.info(f"brake{brake.shape}:{brake}")

                        velocity = ragged_nparray_list_interp(
                            value["list_speed_1s"], ob_num=unit_ob_num
                        )
                        self.logger.info(f"velocity{velocity.shape}:{velocity}")

                        gears = ragged_nparray_list_interp(
                            value["list_gears"], ob_num=unit_gear_num
                        )
                        # upsample gears from 2Hz to 50Hz
                        gears = np.repeat(gears, (signal_freq // gear_freq), axis=1)
                        self.logger.info(f"gears{gears.shape}:{gears}")

                        observation = np.c_[
                            timestamps.reshape((-1, 1)),
                            velocity.reshape(-1, 1),
                            thrust.reshape(-1, 1),
                            brake.reshape(-1, 1),
                            gears.reshape(-1, 1),
                            current.reshape(-1, 1),
                            voltage.reshape(-1, 1),
                        ]  # 1 + 4 + 2 : in
                        self.logger.info(
                            f"observation{observation.shape}:{observation}"
                        )

                    else:
                        self.logger.info(
                            f"show status: {key}:{value}",
                            extra=self.dict_logger,
                        )
                        print(f"{key}:{value}")
            except Exception as X:
                print(f"{X}:data corrupt!")
                self.logger.error(
                    f"show status: exception {X}, data corruption",
                    extra=self.dict_logger,
                )
                return
        else:
            print("upload corrupt!")
            print("reson", remotecan_data)

    def native_send(self):
        # # map2d = [[i * 10 + j for j in range(17)] for i in range(5)]

        # flashing 5 rows of the calibration table
        k0 = 0
        N0 = 5
        # map2d_5rows = self.vcu_calib_table_default.iloc[k0 : k0 + N0, :]  # explicit indexing
        minvel = 0
        maxvel = 30
        map2d_5rows = self.vcu_calib_table_default.loc[
            minvel:maxvel
        ]  # explicit indexing
        self.logger.info(
            f"start sending torque map: from {minvel} kmph to the {maxvel} kmph row.",
            extra=self.dict_logger,
        )
        returncode, ret_str = self.client.send_torque_map(
            pedalmap=map2d_5rows, swap=False
        )
        self.logger.info(
            f"finish sending torque map from {minvel} kmph to the {maxvel} kmph row.: returncode={returncode}, ret_str={ret_str}.",
            extra=self.dict_logger,
        )

        self.logger.info(
            f"start sending torque map: {N0} rows from row {k0} .",
            extra=self.dict_logger,
        )
        returncode, ret_str = self.client.send_torque_map(
            pedalmap=map2d_5rows, swap=True
        )
        self.logger.info(
            f"finish sending torque map {N0} rows from row {k0} with buffer switch: returncode={returncode}, ret_str={ret_str}.",
            extra=self.dict_logger,
        )

        # flashing 6 rows of the calibration table
        k0 = 2
        N0 = 8
        map2d_5rows = self.vcu_calib_table_default.iloc[k0 : k0 + N0, :]
        self.logger.info(
            f"start sending torque map: from {k0}th to the {k0+N0-1}th row.",
            extra=self.dict_logger,
        )
        returncode, ret_str = self.client.send_torque_map(
            pedalmap=map2d_5rows, swap=False
        )
        self.logger.info(
            f"finish sending torque map {N0} rows from row {k0} : returncode={returncode}, ret_str={ret_str}.",
            extra=self.dict_logger,
        )

        # flashing the whole calibration table
        map2d = self.vcu_calib_table_default
        self.logger.info(f"start sending torque map.", extra=self.dict_logger)
        returncode, ret_str = self.client.send_torque_map(pedalmap=map2d, swap=False)
        self.logger.info(
            f"finish sending torque map: returncode={returncode}, ret_str={ret_str}",
            extra=self.dict_logger,
        )


if __name__ == "__main__":
    unittest.main(argv=["submodule-remotecan-test"], exit=False)
