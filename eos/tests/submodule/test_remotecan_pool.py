# system import
# 3rd party import
import bson
import datetime
import inspect
import logging
import os
import unittest
import warnings
from datetime import datetime

import numpy as np
from pymongoarrow.monkey import patch_all
# from pymongoarrow.api import Schema
from bson import ObjectId

import pyarrow as pa
import pymongo as pmg
import pymongoarrow as pmga

# local import
# import src.comm.remotecan.remote_can_client.remote_can_client as remote_can_client
from eos import Pool
from eos import RemoteCan, projroot
from eos.comm import generate_vcu_calibration
from eos.utils import ragged_nparray_list_interp
from eos.config import trucks
from eos.utils.exception import TruckIDError

# import ...src.comm.remotecan.remote_can_client.remote_can_client

# ignore DeprecationWarning
warnings.filterwarnings("ignore", message="currentThread", category=DeprecationWarning)
np.warnings.filterwarnings("ignore", category=DeprecationWarning)

patch_all()

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
        self.trucks = trucks
        self.truck_name = 'VB7'

        self.schema = []
        self.record = []
        self.projroot = projroot
        self.logger = logging.getLogger("eostest")
        self.logger.propagate = False
        self.dictLogger = {"user": inspect.currentframe().f_code.co_name}

        self.truck = self.trucks[self.truck_name]
        self.set_logger(projroot)


        # check if the truck is valid
        self.assertEqual(self.truck_name, self.truck.TruckName)

        # try:
        #     self.truck = self.trucks[self.truck_name]
        # except KeyError as e:
        #     self.logger.error(f"{e}. No Truck with name {self.truck_name}", extra=self.dictLogger)
        #     return
        # except Exception as e:
        #     self.logger.error(f"{e}", extra=self.dictLogger)
        #     return

        # # validate truck ID to be "VB7"
        # try:
        #     if self.truck.TruckName != "VB7":
        #         raise TruckIDError("Truck ID is not VB7")
        # except TruckIDError as e:
        #     self.logger.error(f"Caught Project Exception: {e}", extra=self.dictLogger)
        #     raise e

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
            "test_remotecan_pool-"
            + self.truck.TruckName + '-'
            + datetime.now().isoformat().replace(":", "-")
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

    @unittest.skipIf(site == "internal", "skip for internal test")
    def test_native_pool_deposit_record(self):
        self.logger.info("Start test_pool_deposit", extra=self.dictLogger)
        self.client = RemoteCan(vin=self.truck.VIN)
        self.generate_schemas()
        # test schema[0]
        # self.pool = RecordPool(schema=self.schema[0], username="root", password="Newrizon123",url="mongodb://10.0.64.64:30116/", db_name="record_db", debug=True)
        self.pool = Pool(schema=self.schema[0], db_name="record_db", debug=True)
        self.logger.info("Set client", extra=self.dictLogger)
        self.get_records()
        self.logger.info("Records created.", extra=self.dictLogger)
        for rec in self.record:
            result = self.pool.deposit_record(rec)
            self.logger.info("Quadruple inserted.", extra=self.dictLogger)
            self.assertEqual(result.acknowledged, True)
            self.logger.info(f"Pool has {self.pool.count_records()} records", extra=self.dictLogger)
            rec_inserted = self.pool.find_record_by_id(result.inserted_id)

            self.logger.info("record found.", extra=self.dictLogger)
            self.assertEqual(rec_inserted['timestamp'], rec['timestamp'])
            self.assertEqual(rec_inserted['plot'], rec['plot'])
            self.assertEqual(rec_inserted['observation'], rec['observation'])

        pass
    # @unittest.skipIf(site == "internal", "skip for internal test")
    def test_native_pool_sample_record(self):
        self.client = RemoteCan(vin=self.truck.VIN)
        # self.pool = RecordPool(schema=self.schema[0], username="root", password="Newrizon123",url="mongodb://10.0.64.64:30116/", db_name="record_db", debug=True)
        self.pool = Pool(schema=self.schema[0], db_name="record_db", debug=True)
        self.logger.info("Set client", extra=self.dictLogger)

        self.logger.info("Start creating record pool", extra=self.dictLogger)
        self.create_record_pool(pool_size=128)

        batch_size = 16

        minibatch = self.pool.sample_batch_record(batch_size=batch_size)

        self.logger.info("Start test_pool_sample", extra=self.dictLogger)

    def generate_schemas(self):
        # current state
        self.schema.append({
            '_id': ObjectId,
            "timestamp": datetime,
            'plot': {
                'character': str,
                'when': datetime,
                'where': str
            },
            'observation': [float]
        })

        self.schema.append({
            '_id': ObjectId,
            "timestamp": datetime,
            'plot': {
                'character': str,
                'when': datetime,
                'where': str,
            },
            'observation': {
                'timestamps': datetime,
                'state': {
                    'velocity': [float],
                    'thrust': [float],
                    'brake': [float]
                },
                'action': [float],
                'reward': float,
                'next_state': {
                    'velocity': [float],
                    'thrust': [float],
                    'brake': [float]
                }
            }
        })

        self.schema.append({
            '_id': ObjectId,
            "timestamp": datetime,
            'plot': {
                'character': str,
                'when': datetime,
                'where': str
            },
            'observation': {
                'timestamps': datetime,
                'state': [float],  # [(velocity, thrust, brake)]
                'action': [float], # [row0, row1, row2, row3, row4]
                'reward': float,
                'next_state': [float] # [(velocity, thrust, brake)]
            }
        })

    def get_records(self):
        # current state
        self.native_get()
        out = np.split(self.observation, [1, 4, 5], axis=1) # split by empty string
        (timestamp0, motion_states0, gear_states0, power_states0) = [np.squeeze(e) for e in out]
        ui_sum0 = np.sum(np.prod(power_states0, axis=1)) / 3600.0 * 0.02 # convert to Wh
        self.record.append({
            'timestamp': datetime.fromtimestamp(timestamp0[0]/1000.0),
            'plot': {
                'character': self.truck.TruckName,
                'when': datetime.fromtimestamp(timestamp0[0]/1000.0),
                'where': 'campus',
            },
            'observation': self.observation.tolist()
        })

        # action
        k0 = 0
        N0 = 5
        map2d_5rows = self.vcu_calib_table_default[k0 : k0 + N0, :].reshape(-1).tolist()
        self.logger.info(
            f"Create torque map: from {k0}th to the {k0+N0-1}th row.",
            extra=self.dictLogger,
        )

        # next state
        self.native_get()
        out = [np.squeeze(e) for e in np.split(self.observation, [1, 4, 5], axis=1)] # split by empty string
        (timestamp1, motion_states1, gear_states1, power_states1) = [np.squeeze(e) for e in out]
        ui_sum1 = np.sum(np.prod(power_states1, axis=1)) / 3600.0 * 0.02 # convert to Wh

        cycle_reward = ui_sum1 + ui_sum0

        self.record.append({
            'timestamp':  datetime.fromtimestamp(timestamp0[0]/1000.0),
             'plot': {
                 'character': self.truck.TruckName,
                 'when': datetime.fromtimestamp(timestamp0[0]/1000.0),
                 'where': 'campus',
             },
             'observation': {
                 'state': {
                     'velocity': motion_states0[:,0].tolist(),
                     'thrust': motion_states0[:,1].tolist(),
                     'brake': motion_states0[:,2].tolist(),
                 },
                 'action': map2d_5rows,
                 'reward': cycle_reward,
                 'next_state': {
                     'velocity': motion_states1[:,0].tolist(),
                     'thrust': motion_states1[:,1].tolist(),
                     'brake': motion_states1[:,2].tolist(),
                 },
             }
        })
        self.record.append({
            'timestamp': datetime.fromtimestamp(timestamp0[0]/1000.0),
            'plot': {
                'character': self.truck.TruckName,
                'when': datetime.fromtimestamp(timestamp0[0]/1000.0),
                'where': 'campus',
            },
            'observation': {
                'state': motion_states0.tolist(),
                'action': map2d_5rows,
                'reward': cycle_reward,
                'next_state': motion_states1.tolist(),
            }
        })

    # motion_states0.tolist(), map2d_5rows, cycle_reward, motion_states1.tolist())

    def create_record_pool(self, pool_size=128):
        self.client = RemoteCan(vin=self.truck.VIN)
        self.pool = Pool(url="mongodb://10.0.64.64:30116/", db_name="eos_db", debug=False)
        self.logger.info("Set client", extra=self.dictLogger)

        for i in range(pool_size):
            quadruple = self.get_quadruple()
            self.pool.deposit_record(vin=self.truck.VIN, tuple=quadruple)


    @unittest.skipIf(site == "internal", "skip for internal test")
    def test_native_send(self):
        self.logger.info("Start test_native_send", extra=self.dictLogger)
        self.client = RemoteCan(vin=self.truck.VIN)
        self.logger.info("Set client", extra=self.dictLogger)

        self.native_send()

    def native_get(self):

        signal_success, remotecan_data = self.client.get_signals(
            duration=self.truck.CloudUnitNumber
        )
        self.logger.info(
            f"get_signal(), return state:{signal_success}", extra=self.dictLogger
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

                self.logger.info("show remotecan_data", extra=self.dictLogger)
                signal_freq = self.truck.CloudSignalFrequency
                gear_freq = self.truck.CloudGearFrequency
                unit_duration = self.truck.CloudUnitDuration
                unit_ob_num = unit_duration * signal_freq
                unit_gear_num = unit_duration * gear_freq
                unit_num = self.truck.CloudUnitNumber
                timestamp_upsample_rate = self.truck.CloudSignalFrequency * self.truck.CloudUnitDuration
                # timestamp_num = int(self.observe_length // duration)

                for key, value in remotecan_data.items():
                    if key == "result":
                        self.logger.info("show result", extra=self.dictLogger)

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
                            np.array(timestamps).astype("datetime64[ms]").astype("int")  # convert to int
                        )
                        if len(timestamps_units) != unit_num:
                            raise ValueError(
                                f"timestamps_units length is {len(timestamps_units)}, not {unit_num}"
                            )
                        # upsample gears from 2Hz to 50Hz
                        timestamps_seconds = list(timestamps_units)  # in ms
                        sampling_interval = 1.0 / signal_freq * 1000  # in ms
                        timestamps = [i + j*sampling_interval for i in timestamps_seconds for j in np.arange(unit_ob_num)]
                        timestamps = np.array(timestamps).reshape((self.truck.CloudUnitNumber, -1))


                        # current = np.array(value["list_current_1s"])
                        current = ragged_nparray_list_interp(
                            value["list_current_1s"], ob_num=unit_ob_num
                        )
                        voltage = ragged_nparray_list_interp(
                            value["list_voltage_1s"], ob_num=unit_ob_num
                        )
                        r_v, c_v = voltage.shape
                        # voltage needs to be upsampled in columns if its sample rate is half of the current
                        if c_v == current.shape[1] // 2:
                            voltage = np.repeat(voltage, 2, axis=1)
                        thrust = ragged_nparray_list_interp(
                            value["list_pedal_1s"], ob_num=unit_ob_num
                        )
                        brake = ragged_nparray_list_interp(
                            value["list_brake_pressure_1s"], ob_num=unit_ob_num
                        )
                        velocity = ragged_nparray_list_interp(
                            value["list_speed_1s"], ob_num=unit_ob_num
                        )
                        gears = ragged_nparray_list_interp(
                            value["list_gears"], ob_num=unit_gear_num
                        )
                        # upsample gears from 2Hz to 50Hz
                        gears = np.repeat(gears, (signal_freq // gear_freq), axis=1)
                        self.observation = np.c_[
                            timestamps.reshape((-1, 1)),
                            velocity.reshape(-1, 1),
                            thrust.reshape(-1, 1),
                            brake.reshape(-1, 1),
                            gears.reshape(-1, 1),
                            current.reshape(-1, 1),
                            voltage.reshape(-1, 1),
                        ]  # 1 + 4 + 2
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
            f"finish sending torque map {N0} rows from row {k0} : returncode={returncode}.",
            extra=self.dictLogger,
        )

        self.logger.info(
            f"start sending torque map: from {k0}th to the {k0+N0-1}th row.",
            extra=self.dictLogger,
        )
        returncode = self.client.send_torque_map(
            pedalmap=map2d_5rows, k=k0, N=N0, abswitch=True
        )
        self.logger.info(
            f"finish sending torque map {N0} rows from row {k0} with buffer switch: returncode={returncode}.",
            extra=self.dictLogger,
        )

        # flashing 5 rows of the calibration table
        k0 = 2
        N0 = 8
        map2d_5rows = self.vcu_calib_table_default[k0 : k0 + N0, :].reshape(-1).tolist()
        self.logger.info(
            f"start sending torque map: from {k0}th to the {k0+N0-1}th row.",
            extra=self.dictLogger,
        )
        returncode = self.client.send_torque_map(
            pedalmap=map2d_5rows, k=k0, N=N0, abswitch=False
        )
        self.logger.info(
            f"finish sending torque map {N0} rows from row {k0} : returncode={returncode}.",
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
    unittest.main(argv=["submodule-remotecan-pool-test"], exit=False)
