# system import
# 3rd party import
import unittest
import json
# local import
import src.comm.remotecan.remote_can_client.remote_can_client as remote_can_client

# import ...src.comm.remotecan.remote_can_client.remote_can_client


class TestRemoteCan(unittest.TestCase):
	"""Tests for 'remote_can_client.py'."""
	def test_proxy(self):
		proxies = {
			"http": "http://127.0.0.1:10809",
			"https": "http://127.0.0.1:10809",
		}
		client = remote_can_client.RemoteCan(
			vin="987654321654321M4",
			proxies=proxies)

		map2d = [[i * 10 + j for j in range(17)] for i in range(21)]
		success, reson = client.send_torque_map(map2d)
		if success:
			signal_success, json_ret = client.get_signals(duration=2)
			if signal_success is True:
				try:
					print("print whole json string:")
					json_string = json.dumps(json_ret, indent=4, sort_keys=True, separators=(',', ': '))
					print(json_string)

					for key, value in json_ret.items():
						if key == 'result':
							current = float(value['list_current_ls_zoomed'])
							print("current:")
							print(current)
							voltage = float(value['list_voltage_ls'])
							print("voltage:")
							print(voltage)
							accl = float(value['list_pedal_ls'])
							print("accl")
							print(accl)
							brake = float(value['list_brake_pressure_ls'])
							print("brake:")
							print(brake)
							velocity = float(value['list_speed_ls'])
							print("velocity:")
							print(velocity)
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


	def test_native(self):
		client = remote_can_client.RemoteCan(vin="987654321654321M4")

		map2d = [[i * 10 + j for j in range(17)] for i in range(21)]
		success, reson = client.send_torque_map(map2d)
		if success:
			signal_success, json_ret = client.get_signals(duration=2)
			if signal_success is True:
				try:
					print("print whole json string:")
					json_string = json.dumps(json_ret, indent=4, sort_keys=True, separators=(',', ': '))
					print(json_string)

					for key, value in json_ret.items():
						if key == 'result':
							current = float(value['list_current_ls_zoomed'])
							print("current:")
							print(current)
							voltage = float(value['list_voltage_ls'])
							print("voltage:")
							print(voltage)
							accl = float(value['list_pedal_ls'])
							print("accl")
							print(accl)
							brake = float(value['list_brake_pressure_ls'])
							print("brake:")
							print(brake)
							velocity = float(value['list_speed_ls'])
							print("velocity:")
							print(velocity)
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
