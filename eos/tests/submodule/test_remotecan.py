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


class TestRemoteCan(unittest.TestCase):
    """Tests for 'remote_can_client.py'."""

    site = "internal"

    @unittest.skipIf(site == "internal", "skip for internal test")
    def test_proxy(self):
        proxies = {
            "http": "http://127.0.0.1:20171",
            "https": "http://127.0.0.1:20171",
        }
        proxies_socks = {
            "http": "socks5://127.0.0.1:20170",
            "https": "socks5://127.0.0.1:20170",
        }
        proxies_lantern = {
            "http": "http://127.0.0.1:34663",
            "https": "http://127.0.0.1:34663",
        }
        client = remote_can_client.RemoteCan(
            vin="987654321654321M4", proxies=proxies_lantern
        )

        map2d = [[i * 10 + j for j in range(17)] for i in range(21)]
        success, reson = client.send_torque_map(map2d)
        if success:
            signal_success, json_ret = client.get_signals(duration=2)
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
        client = remote_can_client.RemoteCan(vin="987654321654321M4")
        os.environ["http_proxy"] = ""  # for native test (internal site force no proxy)
        map2d = [[i * 10 + j for j in range(17)] for i in range(21)]
        success, reson = client.send_torque_map(map2d)
        if success:
            signal_success, json_ret = client.get_signals(duration=2)
            if signal_success is True:
                try:
                    # json_string = json.dumps(
                    #     json_ret, indent=4, sort_keys=True, separators=(",", ": ")
                    # )
                    # print(f"print whole json string:{json_string}")

                    for key, value in json_ret.items():
                        if key == "result":

                            with warnings.catch_warnings(record=True) as w:
                                warnings.simplefilter("always")
                                current = np.array(value["list_current_1s_zoomed"])
                                if len(w) > 0:
                                    item_len = [len(item) for item in current]
                                    for count, item in enumerate(current):
                                        item[item_len[count] : item_len.max()] = None

                            print(f"current{current.shape}:{current}")
                            voltage = np.array(value["list_voltage_1s"])
                            print(f"voltage{voltage.shape}:{voltage}")
                            thrust = np.array(value["list_pedal_1s"])
                            print(f"accl{thrust.shape}:{thrust}")
                            brake = np.array(value["list_brake_pressure_1s"])
                            print(f"brake{brake.shape}:{brake}")
                            velocity = np.array(value["list_speed_1s"])
                            print(f"velocity{velocity.shape}:{velocity}")
                            gears = np.array(value["list_gears"])
                            print(f"gears{gears.shape}:{gears}")
                            delay = np.array(value["signal_dalay"])
                            print(f"delay{delay.shape}:{delay}")
                            timestamp = np.array(value["timestamp"])
                            print(
                                f"timestamp{timestamp.shape}:{datetime.fromtimestamp(timestamp.tolist())}"
                            )
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
