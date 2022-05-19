# system import
# 3rd party import
import unittest
import json
import numpy as np
from datetime import datetime
import os
import warnings

# local import
# import src.comm.remotecan.remote_can_client.remote_can_client as remote_can_client
from eos import remote_can_client

# import ...src.comm.remotecan.remote_can_client.remote_can_client

# ignore DeprecationWarning
warnings.filterwarnings("ignore", message='currentThread',category=DeprecationWarning)
np.warnings.filterwarnings('ignore', category=DeprecationWarning)

class TestRemoteCan(unittest.TestCase):
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
        self.client = remote_can_client.RemoteCan(vin="987654321654321M4")

    @unittest.skipIf(site == "internal", "skip for internal test")
    def test_proxy(self):
        self.client = remote_can_client.RemoteCan(
            vin="987654321654321M4", proxies=self.proxies_lantern
        )
        map2d = [[i * 10 + j for j in range(17)] for i in range(21)]
        success, reson = self.client.send_torque_map(map2d)
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
            print("reson", reson)

    def test_native(self):
        map2d = [[i * 10 + j for j in range(17)] for i in range(21)]
        success, reson = self.client.send_torque_map(map2d)
        if success:
            signal_success, json_ret = self.client.get_signals(duration=2)
            if signal_success is True:
                try:
                    # json_string = json.dumps(
                    #     json_ret, indent=4, sort_keys=True, separators=(",", ": ")
                    # )
                    # print(f"print whole json string:{json_string}")

                    for key, value in json_ret.items():
                        if key == "result":
                            # with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.1f}'.format}, linewidth=100):
                            with np.printoptions(suppress=True, linewidth=100):
                                # capture warning about ragged json arrays
                                with np.testing.suppress_warnings() as sup:
                                    log_warning = sup.record(np.VisibleDeprecationWarning, "Creating an ndarray from ragged nested sequences")
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
                                    print(f"gears{gears.shape}:{gears}")

                                    timestamp = np.array(value["timestamp"])
                                    print(f"timestamp{timestamp.shape}:{datetime.fromtimestamp(timestamp.tolist())}")
                        else:
                            print(f"{key}:{value}")
                except Exception as X:
                    print(f"{X}:data corrupt!")
            else:
                print("upload corrupt!")
                print("reson", json_ret)
        else:
            print(f"download corrupt!")
            print("reson", reson)


if __name__ == "__main__":
    unittest.main(argv=["submodule-remotecan-test"], exit=False)
