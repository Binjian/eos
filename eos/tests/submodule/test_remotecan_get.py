# system import
# 3rd party import
import datetime
import inspect
import logging
import os
import unittest
import warnings

import numpy as np

# local import
# import src.comm.remotecan.remote_can_client.remote_can_client as remote_can_client
from eos import RemoteCan, projroot
from eos.comm import generate_vcu_calibration
from eos.utils import ragged_nparray_list_interp
from eos.config import trucks
from eos.utils.exception import TruckIDError

# import ...src.comm.remotecan.remote_can_client.remote_can_client

# ignore DeprecationWarning
warnings.filterwarnings("ignore", message="currentThread", category=DeprecationWarning)
np.warnings.filterwarnings("ignore", category=DeprecationWarning)


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
        self.trucks = trucks
        self.truck_ind = 0  # index of truck to test, 0 is VB7, 1 is VB6, 2 is HQ

        self.projroot = projroot
        self.logger = logging.getLogger("eostest")
        self.logger.propagate = False
        self.dictLogger = {"user": inspect.currentframe().f_code.co_name}
        self.set_logger(projroot)
        self.truck = self.trucks[self.truck_ind]

        # validate truck ID to be "VB7"
        try:
            if self.truck.TruckName != "VB7":
                raise TruckIDError("Truck ID is not VB7")
        except TruckIDError as e:
            self.logger.error(f"Caught Project Exception: {e}", extra=self.dictLogger)
            raise e

        self.observe_length = self.truck.CloudUnitNumber  # number of cloud units 5s

        self.vcu_calib_table_default = generate_vcu_calibration(
            self.truck.PedalScale,
            self.truck.PedalRange,
            self.truck.VelocityScale,
            self.truck.VelocityRange,
            2,
            self.projroot.joinpath("eos/config"),
        )

    def set_logger(self, projroot):
        logroot = projroot.joinpath("data/scratch/tests")
        try:
            os.makedirs(logroot)
        except FileExistsError:
            pass
        logfile = logroot.joinpath(
            "test_remotecan_get-"
            + self.trucks[self.truck_ind].Name
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
    #     self.logger.info("start test_proxy", extra=self.dictLogger)
    #     self.client = RemoteCan(
    #         vin=self.trucks[self.truck_ind].VIN, proxies=self.proxies_lantern
    #     )
    #     self.native_get()
    #
    # @unittest.skipIf(site == "internal", "skip for internal test")
    # def test_proxy_send(self):
    #
    #     self.logger.info("start test_proxy", extra=self.dictLogger)
    #     self.client = RemoteCan(
    #         vin=self.trucks[self.truck_ind].VIN, proxies=self.proxies_lantern
    #     )
    #     self.native_send()

    def test_native_get(self):
        self.logger.info("Start test_native_get", extra=self.dictLogger)
        self.client = RemoteCan(vin=self.truck.VIN)
        self.logger.info("Set client", extra=self.dictLogger)
        self.native_get()

    # @unittest.skipIf(site == "internal", "skip for internal test")
    # def test_native_send(self):
    #     self.logger.info("Start test_native_send", extra=self.dictLogger)
    #     self.client = RemoteCan(vin=self.truck.VIN)
    #     self.logger.info("Set client", extra=self.dictLogger)
    #
    #     self.native_send()

    def native_get(self):

        signal_success, remotecan_data = self.client.get_signals(
            duration=self.observe_length
        )
        self.logger.info(
            f"get_signal(), return state:{signal_success}", extra=self.dictLogger
        )

        data_type = type(remotecan_data)
        self.logger.info(f"data type: {data_type}")
        if not isinstance(remotecan_data, dict):
            raise TypeError("udp sending wrong data type!")
        if signal_success is True:
            try:
                # json_string = json.dumps(
                #     json_ret, indent=4, sort_keys=True, separators=(",", ": ")
                # )
                # print(f"print whole json string:{json_string}")

                self.logger.info("show remotecan_data", extra=self.dictLogger)
                signal_freq = self.truck.CloudSignalFrequency
                gear_freq = self.truck.CloudGearFrequency
                unit_duration = self.truck.CloudUnitDuration
                unit_ob_num = unit_duration * signal_freq
                unit_gear_num = unit_duration * gear_freq
                # timestamp_num = int(self.observe_length // duration)

                for key, value in remotecan_data.items():
                    if key == "result":
                        self.logger.info("show result", extra=self.dictLogger)
                        # current = np.array(value["list_current_1s"])
                        current = value["list_current_1s"]
                        current = ragged_nparray_list_interp(
                            current, ob_num=unit_ob_num
                        )
                        print(f"current{current.shape}:{current}")

                        voltage = ragged_nparray_list_interp(
                            value["list_voltage_1s"], ob_num=unit_ob_num
                        )
                        r_v, c_v = voltage.shape
                        # voltage needs to be upsampled in columns if its sample rate is half of the current
                        if c_v == current.shape[1] // 2:
                            voltage = np.repeat(voltage, 2, axis=1)
                        print(f"voltage{voltage.shape}:{voltage}")

                        thrust = ragged_nparray_list_interp(
                            value["list_pedal_1s"], ob_num=unit_ob_num
                        )
                        print(f"accl{thrust.shape}:{thrust}")

                        brake = ragged_nparray_list_interp(
                            value["list_brake_pressure_1s"], ob_num=unit_ob_num
                        )
                        print(f"brake{brake.shape}:{brake}")

                        velocity = ragged_nparray_list_interp(
                            value["list_speed_1s"], ob_num=unit_ob_num
                        )
                        print(f"velocity{velocity.shape}:{velocity}")

                        gears = ragged_nparray_list_interp(
                            value["list_gears"], ob_num=unit_gear_num
                        )
                        # upsample gears from 2Hz to 50Hz
                        gears = np.repeat(gears, (signal_freq // gear_freq), axis=1)
                        print(f"gears{gears.shape}:{gears}")

                        observation = np.c_[
                            velocity.reshape(-1, 1),
                            thrust.reshape(-1, 1),
                            brake.reshape(-1, 1),
                            current.reshape(-1, 1),
                            voltage.reshape(-1, 1),
                            gears.reshape(-1, 1),
                        ]  # 3 +2 +1 : im 5
                        print(f"observation{observation.shape}:{observation}")

                        timestamps = []
                        separators = (
                            "--T::."  # adaption separators of the raw intest string
                        )
                        start_century = "20"
                        for ts in value["timestamps"]:
                            # create standard iso string datetime format
                            ts_substrings = [
                                ts[i : i + 2] for i in range(0, len(ts), 2)
                            ]
                            ts_iso = start_century
                            for i, sep in enumerate(separators):
                                ts_iso = ts_iso + ts_substrings[i] + sep
                            ts_iso = ts_iso + ts_substrings[-1]
                            timestamps.append(ts_iso)
                        timestamps = (
                            np.array(timestamps).astype("datetime64[ms]").astype("int")
                        )
                        print(
                            f"timestamp{timestamps.shape}:{timestamps.astype('datetime64[ms]')}"
                        )
                    else:
                        self.logger.info(
                            f"show status: {key}:{value}", extra=self.dictLogger
                        )
                        print(f"{key}:{value}")
            except Exception as X:
                print(f"{X}:data corrupt!")
                self.logger.error(
                    f"show status: exception {X}, data corruption",
                    extra=self.dictLogger,
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
        map2d_5rows = self.vcu_calib_table_default[k0 : k0 + N0, :].reshape(-1).tolist()
        self.logger.info(
            f"start sending torque map: from {k0}th to the {k0+N0-1}th row.",
            extra=self.dictLogger,
        )
        returncode = self.client.send_torque_map(
            pedalmap=map2d_5rows, k=k0, N=N0, abswitch=False
        )
        self.logger.info(
            f"finish sending torque map: returncode={returncode}.",
            extra=self.dictLogger,
        )

        # flashing the whole calibration table
        map2d = self.vcu_calib_table_default.reshape(-1).tolist()
        self.logger.info(f"start sending torque map.", extra=self.dictLogger)
        returncode = self.client.send_torque_map(
            pedalmap=map2d, k=0, N=14, abswitch=False
        )
        self.logger.info(
            f"finish sending torque map: returncode={returncode}.",
            extra=self.dictLogger,
        )


if __name__ == "__main__":
    unittest.main(argv=["submodule-remotecan-test"], exit=False)
