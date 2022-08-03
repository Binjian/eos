# system import
# 3rd party import
import unittest
import json
import numpy as np
import os
import datetime
import logging
import inspect
import warnings
from collections import namedtuple
from eos.comm import generate_vcu_calibration
from eos import projroot

# local import
# import src.comm.remotecan.remote_can_client.remote_can_client as remote_can_client
from eos import RecordPool

# import ...src.comm.remotecan.remote_can_client.remote_can_client

# ignore DeprecationWarning
warnings.filterwarnings("ignore", message="currentThread", category=DeprecationWarning)
np.warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestRemoteCanPool(unittest.TestCase):
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
        self.TruckType = namedtuple(
            "Truck",
            [
                "Name",
                "VIN",
                "Plate",
                "Maturity",
                "PedalRange",
                "PedalScale",
                "VelocityRange",
                "VelocityScale",
            ],
        )
        self.trucks = [
            self.TruckType(
                Name="VB7",
                VIN="HMZABAAH7MF011058",
                Plate="77777777",
                Maturity="VB",
                PedalRange=[0.0, 1.0],
                PedalScale=17,
                VelocityRange=[0.0, 120],
                VelocityScale=14,
            ),
            self.TruckType(
                Name="VB6",
                VIN="HMZABAAH5MF011057",
                Plate="66666666",
                Maturity="VB",
                PedalRange=[0.0, 1.0],
                PedalScale=17,
                VelocityRange=[0.0, 120],
                VelocityScale=14,
            ),
            self.TruckType(
                Name="HQB",
                VIN="NEWRIZON020220328",
                Plate="00000000",
                Maturity="VB",
                PedalRange=[0.0, 1.0],
                PedalScale=17,
                VelocityRange=[0.0, 120],
                VelocityScale=14,
            ),
        ]  # HQ Bench
        self.truck_ind = 0  # index of truck to test, 0 is VB7, 1 is VB6, 2 is HQ
        self.projroot = projroot
        self.logger = logging.getLogger("__name__")
        self.logger.propagate = False
        self.dictLogger = {"user": inspect.currentframe().f_code.co_name}
        self.set_logger(projroot)

        self.vcu_calib_table_default = generate_vcu_calibration(
            self.trucks[self.truck_ind].PedalScale,
            self.trucks[self.truck_ind].PedalRange,
            self.trucks[self.truck_ind].VelocityScale,
            self.trucks[self.truck_ind].VelocityRange,
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

    @unittest.skipIf(site == "internal", "skip for internal test")
    def test_proxy(self):

        self.client = RemoteCan(
            vin=self.trucks[self.truck_ind].VIN, proxies=self.proxies_lantern
        )
        map2d = [[i * 10 + j for j in range(17)] for i in range(21)]
        success, response = self.client.send_torque_map(map2d)
        if success:
            signal_success, json_ret = self.client.get_signals(duration=2)
            if signal_success is True:
                try:
                    print("print whole json string:")
                    json_string = json.dumps(
                        json_ret, indent=4, sort_keys=True, separators=(",", ": ")
                    )
                    print(json_string)
                except Exception as X:
                    print(f"{X}:data corrupt!")
            else:
                print("upload corrupt!")
                print("reson", json_ret)
        else:
            print(f"download corrupt!")
            print("response:", response)

    def test_native(self):
        self.logger.info("Start test_native", extra=self.dictLogger)
        self.client = RemoteCan(vin=self.trucks[self.truck_ind].VIN)

        self.logger.info("Set client", extra=self.dictLogger)
        signal_success, remotecan_data = self.client.get_signals(duration=2)
        self.logger.info(
            f"get_signal(), return state:{signal_success}", extra=self.dictLogger
        )

        data_type = type(remotecan_data)
        print("data type:", data_type)
        if not isinstance(remotecan_data, dict):
            raise TypeError("udp sending wrong data type!")
        if signal_success is True:
            try:
                # json_string = json.dumps(
                #     json_ret, indent=4, sort_keys=True, separators=(",", ": ")
                # )
                # print(f"print whole json string:{json_string}")

                self.logger.info("show remotecan_data", extra=self.dictLogger)
                for key, value in remotecan_data.items():
                    if key == "result":
                        self.logger.info("show result", extra=self.dictLogger)
                        # with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.1f}'.format}, linewidth=100):
                        with np.printoptions(suppress=True, linewidth=100):
                            # capture warning about ragged json arrays
                            with np.testing.suppress_warnings() as sup:
                                log_warning = sup.record(
                                    np.VisibleDeprecationWarning,
                                    "Creating an ndarray from ragged nested sequences",
                                )
                                current = np.array(value["list_current_1s"])
                                if len(log_warning) > 0:
                                    log_warning.pop()
                                    item_len = [len(item) for item in current]
                                    for count, item in enumerate(current):
                                        item[item_len[count] : max(item_len)] = None
                                print(f"current{current.shape}:{current}")

                                voltage = np.array(value["list_voltage_1s"])
                                if len(log_warning):
                                    log_warning.pop()
                                    item_len = [len(item) for item in voltage]
                                    for count, item in enumerate(voltage):
                                        item[item_len[count] : max(item_len)] = None
                                r_v, c_v = voltage.shape
                                # voltage needs to be upsampled in columns if its sample rate is half of the current
                                if c_v != current.shape[1]:
                                    voltage_upsampled = np.empty(
                                        (r_v, 1, c_v, 2), dtype=voltage.dtype
                                    )
                                    voltage_upsampled[...] = voltage[:, None, :, None]
                                    voltage = voltage_upsampled.reshape(r_v, c_v * 2)
                                print(f"voltage{voltage.shape}:{voltage}")

                                thrust = np.array(value["list_pedal_1s"])
                                if len(log_warning) > 0:
                                    log_warning.pop()
                                    item_len = [len(item) for item in thrust]
                                    for count, item in enumerate(thrust):
                                        item[item_len[count] : max(item_len)] = None
                                print(f"accl{thrust.shape}:{thrust}")

                                brake = np.array(value["list_brake_pressure_1s"])
                                if len(log_warning) > 0:
                                    log_warning.pop()
                                    item_len = [len(item) for item in brake]
                                    for count, item in enumerate(brake):
                                        item[item_len[count] : max(item_len)] = None
                                print(f"brake{brake.shape}:{brake}")

                                velocity = np.array(value["list_speed_1s"])
                                if len(log_warning) > 0:
                                    log_warning.pop()
                                    item_len = [len(item) for item in velocity]
                                    for count, item in enumerate(velocity):
                                        item[item_len[count] : max(item_len)] = None
                                print(f"velocity{velocity.shape}:{velocity}")

                                gears = np.array(value["list_gears"])
                                if len(log_warning) > 0:
                                    log_warning.pop()
                                    item_len = [len(item) for item in gears]
                                    for count, item in enumerate(gears):
                                        item[item_len[count] : max(item_len)] = None
                                # upsample gears from 2Hz to 25Hz
                                r_v, c_v = gears.shape
                                gears_upsampled = np.empty(
                                    (r_v, 1, c_v, 25), dtype=gears.dtype
                                )
                                gears_upsampled[...] = gears[:, None, :, None]
                                gears = gears_upsampled.reshape(r_v, c_v * 25)
                                # gears = np.c_[
                                #     gears, gears[:, -1]
                                # ]  # duplicate last gear on the end
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
                                for ts in value["timestamps"]:
                                    ts_iso = (
                                        "20"
                                        + ts[:2]
                                        + "-"
                                        + ts[2:4]
                                        + "-"
                                        + ts[4:6]
                                        + "T"
                                        + ts[6:8]
                                        + ":"
                                        + ts[8:10]
                                        + ":"
                                        + ts[10:12]
                                        + "."
                                        + ts[12:14]
                                    )
                                    timestamps.append(ts_iso)
                                timestamps = np.array(timestamps).astype(
                                    "datetime64[ms]"
                                )

                                print(f"timestamp{timestamps.shape}:{timestamps}")
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

        # map2d = [[i * 10 + j for j in range(17)] for i in range(5)]
        map2d = self.vcu_calib_table_default.reshape(-1).tolist()
        map2d_5rows = self.vcu_calib_table_default[:5, :].reshape(-1).tolist()
        self.logger.info(f"start sending torque map.", extra=self.dictLogger)
        success, response = self.client.send_torque_map(map2d_5rows)
        self.logger.info(
            f"finish sending torque map: success={success}, response={response}.",
            extra=self.dictLogger,
        )
        if success:
            print("torque map sent")
            print("response", response)
        else:
            print("torque map failed")
            print("response:", response)


if __name__ == "__main__":
    unittest.main(argv=["submodule-remotecan-test"], exit=False)
