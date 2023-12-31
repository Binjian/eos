# system import
# 3rd party import
import datetime
import inspect
import logging
import os
import subprocess
import time
import unittest
import warnings
from datetime import datetime

import bson
import numpy as np
import pyarrow as pa
import pymongo as pmg
import pymongoarrow as pmga

# from pymongoarrow.api import Schema
from bson import ObjectId
from pymongoarrow.monkey import patch_all
from tensorflow.keras.utils import pad_sequences  # type: ignore

from eos import Pool, RemoteCanClient, proj_root
from eos.config import (
    can_servers_by_name,
    db_config_servers_by_name,
    episode_schemas,
    generate_vcu_calibration,
    record_schemas,
    trucks_by_id,
    trucks_by_vin,
)
from eos.data_io.utils import ragged_nparray_list_interp, timestamps_from_can_strings
from eos.data_io.utils.exception import TruckIDError

# local import
# import src.comm.remote


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
        self.trucks_by_id = trucks_by_id
        self.truck_name = "VB7"

        self.db_server_name = "mongo_local"
        self.db_server = db_config_servers_by_name[self.db_server_name]
        self.assertEqual(self.db_server_name, self.db_server.server_name)

        self.can_server_name = "can_intra"
        self.can_server = can_servers_by_name[self.can_server_name]
        self.assertEqual(self.can_server_name, self.can_server.server_name)

        self.record_schemas = record_schemas
        self.episode_schemas = episode_schemas

        self.rec_schema = []
        self.epi_schema = []
        self.record = []
        self.proj_root = proj_root
        self.logger = logging.getLogger("eostest")
        self.logger.propagate = False
        self.dict_logger = {"user": inspect.currentframe().f_code.co_name}

        self.truck = self.trucks_by_id[self.truck_name]
        self.set_logger(proj_root)
        self.logger.info(
            f"Truck: {self.truck.vid}-{self.truck.vin}",
            extra=self.dict_logger,
        )
        self.logger.info(
            f"DB server: {self.db_server.server_name}",
            extra=self.dict_logger,
        )
        self.logger.info(
            f"Can server: {self.can_server.server_name}",
            extra=self.dict_logger,
        )

        # check if the truck is valid
        self.assertEqual(self.truck_name, self.truck.vid)

        self.observe_length = self.truck.CloudUnitNumber  # number of cloud units 5s

        self.vcu_calib_table_default = generate_vcu_calibration(
            self.truck.PedalScale,
            self.truck.PedalRange,
            self.truck.VelocityScale,
            self.truck.VelocityRange,
            2,
            self.proj_root.joinpath("eos/config"),
        )

    def set_logger(self, proj_root):
        log_root = proj_root.joinpath("data/scratch/tests")
        try:
            os.makedirs(log_root)
        except FileExistsError:
            pass
        logfile = log_root.joinpath(
            "test_remotecan_pool-"
            + self.truck.vid
            + "-"
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

    @unittest.skipIf(site == "internal", "skip for internal test")
    def test_native_pool_deposit_episode(self):
        self.logger.info("Start test_pool_deposit", extra=self.dict_logger)
        self.client = RemoteCanClient(
            truck_name=self.truck.vid,
            url="http://" + self.can_server.host + ":" + self.can_server.port + "/",
        )
        self.epi_sch = self.episode_schemas["episode_deep"]

        self.db_server_name = "mongo_local"
        self.db_server = db_config_servers_by_name[self.db_server_name]
        self.assertEqual(self.db_server_name, self.db_server.server_name)
        # self.db = self.db["mongo_local"]
        # self.generate_epi_schemas()
        # test schema[0]
        # self.pool = RecordPool(schema=self.schema[0], username="root", password="Newrizon123",url="mongodb://10.0.64.64:30116/", db_name="record_db", debug=True)
        self.pool = Pool(
            url=self.db_server.host,
            username=self.db_server.user_name,
            password=self.db_server.password,
            schema=self.epi_sch.STRUCTURE,
            db_name=self.db_server.database_name,
            coll_name=self.db_server.Epicollection_name,
            debug=True,
        )
        self.logger.info(
            f"Connected to MongoDB {self.db.database_name}, collection {self.db.Epicollection_name}",
            extra=self.dict_logger,
        )
        self.logger.info("Set client", extra=self.dict_logger)
        self.get_an_episode()
        self.logger.info("An Episode is created.", extra=self.dict_logger)
        self.logger.info("Start deposit an episode", extra=self.dict_logger)

        result = self.pool.deposit_item(self.episode)
        self.logger.info("Episode inserted.", extra=self.dict_logger)
        self.assertEqual(result.acknowledged, True)
        pool_count = self.pool.count_items(
            truck_id=self.truck.vid, driver_id="zheng-longfei"
        )
        self.logger.info(f"Pool has {pool_count} records", extra=self.dict_logger)
        epi_inserted = self.pool.find_item(result.inserted_id)

        self.logger.info("episode found.", extra=self.dict_logger)
        self.assertEqual(epi_inserted["timestamp"], self.episode["timestamp"])
        self.assertEqual(epi_inserted["plot"], self.episode["plot"])
        self.assertEqual(epi_inserted["history"], self.episode["history"])

        self.logger.info("End test deposit records", extra=self.dict_logger)

    @unittest.skipIf(site == "internal", "skip for internal test")
    def test_native_pool_sample_episode(self):
        # coll_name = "episode_coll1"
        # db_name = "test_episode_db"
        self.client = RemoteCanClient(
            truck_name=self.truck.vid,
            url="http://" + self.can_server.host + ":" + self.can_server.port + "/",
        )

        self.epi_sch = self.episode_schemas["episode_deep"]
        self.db_server_name = "mongo_local"
        self.db_server = db_servers_by_name[self.db_server_name]
        self.assertEqual(self.db_server_name, self.db_server.server_name)
        # self.db = self.db["mongo_local"]
        # self.generate_epi_schemas()
        # test schema[0]
        # self.pool = RecordPool(schema=self.schema[0], username="root", password="Newrizon123",url="mongodb://10.0.64.64:30116/", db_name="record_db", debug=True)
        self.pool = Pool(
            url=self.db_server.host,
            username=self.db_server.user_name,
            password=self.db_server.password,
            schema=self.epi_sch.STRUCTURE,
            db_name=self.db_server.database_name,
            coll_name=self.db_server.Epicollection_name,
            debug=True,
        )
        self.logger.info(
            f"Connected to MongoDB {self.db_server.database_name}, collection {self.db_server.Epicollection_name}",
            extra=self.dict_logger,
        )

        dt_start = datetime.fromisoformat(
            "2022-10-28T11:30:00.000"
        )  # start from 2022-01-01T08:00:00.000
        dt_end = datetime.fromisoformat(
            "2022-10-31T11:37:00.000"
        )  # start from 2022-01-01T08:00:00.000
        self.logger.info("start count_times.", extra=self.dict_logger)
        rec_cnt = self.pool.count_items(
            vehicle_id=self.truck.vid,
            driver_id="longfei",
            dt_start=dt_start,
            dt_end=dt_end,
        )
        self.logger.info(f"collection has {rec_cnt} episodes", extra=self.dict_logger)
        # if rec_cnt < 8:
        #     self.logger.info("Start creating record pool", extra=self.dict_logger)
        #     self.add_to_episode_pool(pool_size=8)

        self.logger.info("start test_pool_sample of size 4.", extra=self.dict_logger)
        # batch_4 = self.pool.sample_batch_items(batch_size=4)
        batch_4 = self.pool.sample_batch_items(
            batch_size=4, vehicle_id="VB7", dt_start=dt_start, dt_end=dt_end
        )
        self.logger.info("done test_pool_sample of size 4.", extra=self.dict_logger)
        self.assertEqual(len(batch_4), 4)
        # batch_24 = self.pool.sample_batch_items(batch_size=24)
        self.logger.info("start test_pool_sample of size 30.", extra=self.dict_logger)
        batch_30 = self.pool.sample_batch_items(
            batch_size=30, vehicle_id="VB7", dt_start=dt_start, dt_end=dt_end
        )
        self.logger.info("done test_pool_sample of size 30.", extra=self.dict_logger)
        self.assertEqual(len(batch_30), 30)
        # get dimension of the history
        state_length = (
            batch_4[0]["plot"]["states"]["length"] * self.truck.ObservationNumber
        )
        action_row_number = batch_4[0]["plot"]["actions"]["action_row_number"]
        action_column_number = batch_4[0]["plot"]["actions"]["action_column_number"]
        action_length = action_column_number * action_row_number
        self.logger.info(
            f"state length: {state_length}, action length: {action_length}.",
            extra=self.dict_logger,
        )
        # test codecs
        # reward series
        r_n_t = [
            [history["reward"] for history in episode["history"]]
            for episode in batch_30
        ]
        r_n_t1 = pad_sequences(
            r_n_t,
            padding="post",
            dtype="float32",
            value=-10000,
        )
        self.logger.info("done decoding reward.", extra=self.dict_logger)

        # states series
        o_n_l0 = [
            [history["states"] for history in episode["history"]]
            for episode in batch_30
        ]
        o_n_l1 = [
            [[step[i] for step in state] for state in o_n_l0]
            for i in np.arange(state_length)
        ]  # list (n_obs) of lists (batch_size) of lists with variable observation length

        try:
            o_n_t1 = np.array(
                [
                    pad_sequences(
                        o_n_l1i,
                        padding="post",
                        dtype="float32",
                        value=-10000,
                    )  # return numpy array
                    for o_n_l1i in o_n_l1
                ]  # return numpy array list
            )  # return numpy array list of size (n_obs, batch_size, max(len(o_n_l1i))),
            # max(len(o_n_l1i)) is the longest sequence in the batch, should be the same for all observations
            # otherwise observation is ragged, throw exception
            o_n_t2 = o_n_t1.transpose(
                (1, 2, 0)
            )  # return numpy array list of size (batch_size,max(len(o_n_l1i)), n_obs)
        except:
            self.logger.error(
                "Ragged observation state o_n_l1!", extra=self.dict_logger
            )
        # logger.info(f"o_n_t.shape: {self.o_n_t.shape}")
        self.logger.info("done decoding states.", extra=self.dict_logger)

        # starting row series, not used for now
        a_n_start_t = [
            [history["action_start_row"] for history in episode["history"]]
            for episode in batch_30
        ]
        a_n_start_t1 = pad_sequences(
            a_n_start_t,
            padding="post",
            dtype="float32",
            value=-10000,
        )
        self.logger.info("done decoding starting row.", extra=self.dict_logger)

        a_n_l0 = [
            [history["actions"] for history in episode["history"]]
            for episode in batch_30
        ]
        a_n_l1 = [
            [[step[i] for step in act] for act in a_n_l0]
            for i in np.arange(action_length)
        ]  # list (n_act) of lists (batch_size) of lists with variable observation length

        try:
            a_n_t1 = np.array(
                [
                    pad_sequences(
                        a_n_l1i,
                        padding="post",
                        dtype="float32",
                        value=-10000,
                    )  # return numpy array
                    for a_n_l1i in a_n_l1
                ]  # return numpy array list
            )  # return numpy array list of size (n_obs, batch_size, max(len(o_n_l1i))),
            # max(len(o_n_l1i)) is the longest sequence in the batch, should be the same for all observations
            # otherwise observation is ragged, throw exception
            a_n_t2 = a_n_t1.transpose(
                (1, 2, 0)
            )  # return numpy array list of size (batch_size,max(len(o_n_l1i)), n_obs)
        except:
            self.logger.error("Ragged action state a_n_l1!", extra=self.dict_logger)
        self.logger.info("done decoding actions.", extra=self.dict_logger)

    @unittest.skipIf(site == "internal", "skip for internal test")
    def test_native_pool_sample_record(self):
        # self.client = RemoteCanClient(
        #     truck_name=self.truck.vid,
        #     url="http://" + self.can_server.host+ ":" + self.can_server.port + "/",
        # )
        # self.generate_record_schemas()
        self.rec_sch = self.record_schemas["record_deep"]
        self.db_server_name = "mongo_local"
        self.db_server = db_servers_by_name[self.db_server_name]
        self.assertEqual(self.db_server_name, self.db_server.server_name)
        # self.generate_record_schemas()
        # test schema[0]
        # self.pool = RecordPool(schema=self.schema[0], username="root", password="Newrizon123",url="mongodb://10.0.64.64:30116/", db_name="record_db", debug=True)
        self.pool = Pool(
            url=self.db_server.host,
            username=self.db_server.user_name,
            password=self.db_server.password,
            schema=self.rec_sch.STRUCTURE,
            db_name=self.db_server.database_name,
            coll_name=self.db_server.Reccollection_name,
            debug=True,
        )
        self.logger.info("Set client and pool", extra=self.dict_logger)

        rec_cnt = self.pool.count_items(
            vehicle_id=self.truck.vid, driver_id="zheng-longfei"
        )
        # if rec_cnt < 4:
        #     self.logger.info("Start creating record pool", extra=self.dict_logger)
        #     self.add_to_record_pool(pool_size=16)

        self.logger.info("start test_pool_sample of size 4.", extra=self.dict_logger)

        dt_start = datetime.fromisoformat(
            "2022-10-25T11:30:00.000"
        )  # start from 2022-01-01T08:00:00.000
        dt_end = datetime.fromisoformat(
            "2022-10-25T11:37:00.000"
        )  # start from 2022-01-01T08:00:00.000
        batch_4 = self.pool.sample_batch_items(
            batch_size=4, vehicle_id="VB7", dt_start=dt_start, dt_end=dt_end
        )
        self.logger.info("done test_pool_sample of size 4.", extra=self.dict_logger)
        self.assertEqual(len(batch_4), 4)
        batch_24 = self.pool.sample_batch_items(
            batch_size=24, vehicle_id="VB7", dt_start=dt_start, dt_end=dt_end
        )
        self.logger.info("done test_pool_sample of size 24.", extra=self.dict_logger)
        self.assertEqual(len(batch_24), 24)
        batch_64 = self.pool.sample_batch_items(
            batch_size=64, vehicle_id="VB7", dt_start=dt_start, dt_end=dt_end
        )
        self.logger.info("done test_pool_sample of size 64.", extra=self.dict_logger)
        self.assertEqual(len(batch_64), 64)

        # test decoding
        state = [rec["observation"]["state"] for rec in batch_24]
        action = [rec["observation"]["action"] for rec in batch_24]
        reward = [rec["observation"]["reward"] for rec in batch_24]
        next_state = [rec["observation"]["next_state"] for rec in batch_24]

    @unittest.skipIf(site == "internal", "skip for internal test")
    def test_native_pool_deposit_record(self):
        self.logger.info("Start test_pool_deposit", extra=self.dict_logger)
        self.client = RemoteCanClient(
            truck_name=self.truck.vid,
            url="http://" + self.can_server.host + ":" + self.can_server.port + "/",
        )
        # self.generate_record_schemas()
        self.rec_sch = self.record_schemas["record_deep"]
        # test schema[0]
        self.db_server_name = "mongo_local"
        self.db_server = db_servers_by_name[self.db_server_name]
        self.assertEqual(self.db_server_name, self.db_server.server_name)
        self.pool = Pool(
            url=self.db_server.host,
            username=self.db_server.user_name,
            password=self.db_server.password,
            schema=self.epi_sch.STRUCTURE,
            db_name=self.db_server.database_name,
            coll_name=self.db_server.Reccollection_name,
            debug=True,
        )
        self.logger.info("Set client and pool", extra=self.dict_logger)
        self.get_records()
        self.logger.info("Records created.", extra=self.dict_logger)
        self.logger.info("Start deposit records", extra=self.dict_logger)
        for rec in self.record:
            result = self.pool.deposit_item(rec)
            self.logger.info("Record inserted.", extra=self.dict_logger)
            self.assertEqual(result.acknowledged, True)
            rec_cnt = self.pool.count_items(
                vehicle_id=self.truck.vid, driver_id="zheng-longfei"
            )
            self.logger.info(f"Pool has {rec_cnt} records", extra=self.dict_logger)
            rec_inserted = self.pool.find_item(result.inserted_id)

            self.logger.info("record found.", extra=self.dict_logger)
            self.assertEqual(rec_inserted["timestamp"], rec["timestamp"])
            self.assertEqual(rec_inserted["plot"], rec["plot"])
            self.assertEqual(rec_inserted["observation"], rec["observation"])

        self.logger.info("End test deposit redords", extra=self.dict_logger)

    # @unittest.skipIf(site == "internal", "skip for internal test")
    def test_native_pool_consecutive_observations(self):
        self.client = RemoteCanClient(
            truck_name=self.truck.vid,
            url="http://" + self.can_server.host + ":" + self.can_server.port + "/",
        )

        # hostip = self.can_server.Url
        # response = os.system("ping -c 1 " + hostip)
        # if response == 0:
        #     self.logger.info(f"{hostip} is up!", extra=self.dict_logger)
        # else:
        #     self.logger.info(f"{hostip} is down!", extra=self.dict_logger)
        # # response_telnet = os.system(f"curl -v telnet://{hostname}")
        # # self.logger.info(
        # #     f"Telnet {hostname} response: {response_telnet}!", extra=self.dict_logger
        # # )

        # response = os.system("ping -c 1 " + self.can_server.Url)
        try:
            response_ping = subprocess.check_output(
                "ping -c 1 " + self.can_server.host, shell=True, timeout=1
            )
        except subprocess.CalledProcessError as e:
            self.logger.info(
                f"{self.can_server.host} is down, responds: {response_ping}"
                f"return code: {e.returncode}, output: {e.output}!",
                extra=self.dict_logger,
            )
        self.logger.info(
            f"{self.can_server.host} is up, responds: {response_ping}!",
            extra=self.dict_logger,
        )

        try:
            response_telnet = subprocess.check_output(
                f"timeout 1 telnet {self.can_server.host} {self.can_server.port}",
                shell=True,
            )
            self.logger.info(
                f"Telnet {self.can_server.host} responds: {response_telnet}!",
                extra=self.dict_logger,
            )
        except subprocess.CalledProcessError as e:
            self.logger.info(
                f"{self.can_server.host} return code: {e.returncode}, output: {e.output}!",
                extra=self.dict_logger,
            )
        except subprocess.TimeoutExpired as e:
            self.logger.info(
                f"{self.can_server.host} timeout. cmd: {e.cmd}, output: {e.output}, timeout: {e.timeout}!",
                extra=self.dict_logger,
            )

            self.logger.info(
                f"{self.can_server.host} is up, responds: {response_telnet}!",
                extra=self.dict_logger,
            )

        self.logger.info("Start observation test", extra=self.dict_logger)
        for rec_cnt in range(3):
            self.native_get()
            self.logger.info(
                f"Get and deposit Observation No. {rec_cnt}",
                extra=self.dict_logger,
            )
            time.sleep(2)

        self.logger.info(
            "Done with get consecutive observation test", extra=self.dict_logger
        )

    @unittest.skipIf(site == "internal", "skip for internal test")
    def test_native_pool_consecutive_records(self):
        self.client = RemoteCanClient(
            truck_name=self.truck.vid,
            url="http://" + self.can_server.host + ":" + self.can_server.port + "/",
        )
        # self.generate_record_schemas()
        self.rec_sch = self.record_schemas["record_deep"]
        self.db_server_name = "mongo_local"
        self.db_server = db_servers_by_name[self.db_server_name]
        self.assertEqual(self.db_server_name, self.db_server.server_name)
        self.pool = Pool(
            url=self.db_server.host,
            username=self.db_server.user_name,
            password=self.db_server.password,
            schema=self.epi_sch.STRUCTURE,
            db_name=self.db_server.database_name,
            coll_name=self.db_server.Reccollection_name,
            debug=True,
        )
        self.logger.info("Set client and pool", extra=self.dict_logger)

        rec_count = self.pool.count_items(
            truck_id=self.truck.vid, driver_id="zheng-longfei"
        )
        self.logger.info(
            f"Start observation test wth {rec_count} records",
            extra=self.dict_logger,
        )
        for rec_cnt in range(16):
            self.get_ddpg_record()
            self.pool.deposit_item(self.ddpg_record)
            self.logger.info(
                f"Get and deposit Observation No. {rec_cnt}",
                extra=self.dict_logger,
            )
            # self.native_get()
            # out = np.split(self.observation, [1, 4, 5], axis=1)  # split by empty string
            # (timestamp0, motion_states0, gear_states0, power_states0) = [
            #     np.squeeze(e) for e in out
            # ]
            # self.logger.info(f"Get Observation No. {rec_cnt}", extra=self.dict_logger)

        self.logger.info(
            "Done with consecutive getting record test", extra=self.dict_logger
        )

    # @unittest.skipIf(site == "internal", "skip for internal test")
    def test_native_pool_consecutive_flash_test(self):
        self.client = RemoteCanClient(
            truck_name=self.truck.vid,
            url="http://" + self.can_server.host + ":" + self.can_server.port + "/",
        )
        # self.generate_record_schemas()
        hostip = self.can_server.host
        response = os.system("ping -c 1 " + hostip)
        if response == 0:
            self.logger.info(f"{hostip} is up!", extra=self.dict_logger)
        else:
            self.logger.info(f"{hostip} is down!", extra=self.dict_logger)
        # response_telnet = os.system(f"curl -v telnet://{hostname}")
        # self.logger.info(
        #     f"Telnet {hostname} response: {response_telnet}!", extra=self.dict_logger
        # )

        # self.rec_sch = self.record_schemas["record_deep"]
        # self.db_server_name = "mongo_local"
        # self.db_server = db_servers[self.db_server_name]
        # self.assertEqual(self.db_server_name, self.db_server.server_name)
        # self.pool = Pool(
        #     url=self.db_server.Url,
        #     username=self.db_server.user_name,
        #     password=self.db_server.password,
        #     schema=self.epi_sch.STRUCTURE,
        #     db_name=self.db_server.database_name,
        #     coll_name=self.db_server.Reccollection_name,
        #     debug=True,
        # )
        # self.logger.info("Set client and pool", extra=self.dict_logger)

        # map2d = self.vcu_calib_table_default
        # self.logger.info(f"start sending torque map.", extra=self.dict_logger)
        # returncode, ret_str = self.client.send_torque_map(pedalmap=map2d, swap=False)
        # self.logger.info(
        #     f"finish sending torque map: returncode={returncode}.",
        #     extra=self.dict_logger,
        # )

        self.logger.info("Start consecutive flashing test", extra=self.dict_logger)
        for rec_cnt in range(2):
            self.native_send()
            time.sleep(0.5)

        self.logger.info("Done with flashing test", extra=self.dict_logger)

        # # flashing the whole calibration table
        # map2d = self.vcu_calib_table_default
        # self.logger.info(f"start sending torque map.", extra=self.dict_logger)
        # returncode, ret_str = self.client.send_torque_map(pedalmap=map2d, swap=False)
        # self.logger.info(
        #     f"finish sending torque map: returncode={returncode}.",
        #     extra=self.dict_logger,
        # )

    def native_send(self):
        # flashing 5 rows of the calibration table
        k0 = 2
        N0 = 4
        map2d_5rows = self.vcu_calib_table_default.iloc[k0 : k0 + N0, :]
        self.logger.info(
            f"start sending torque map: from {k0}th to the {k0+N0-1}th row.",
            extra=self.dict_logger,
        )
        timeout = N0 + 9
        returncode, ret_str = self.client.send_torque_map(
            pedalmap=map2d_5rows, swap=False, timeout=timeout
        )
        self.logger.info(
            f"finish sending torque map {N0} rows from row {k0} : returncode={returncode}, ret_str={ret_str}.",
            extra=self.dict_logger,
        )

    def generate_epi_schemas(self):
        # current state
        self.epi_schema.append(
            {
                "_id": ObjectId,
                "timestamp": datetime,
                "plot": {
                    "character": str,
                    "when": datetime,
                    "where": str,
                    "length": int,
                    "states": {
                        "velocity_unit_code": "kmph",
                        "thrust_unit_code": "percentage",
                        "brake_unit_code": "percentage",
                        "length": int,
                    },
                    "actions": {
                        "action_row_number": int,
                        "action_column_number": int,
                        "action_start_row": int,
                    },
                    "reward": {
                        "reward_unit": "wh",
                    },
                },
                "history": [float],
            }
        )

        self.epi_schema.append(
            {
                "_id": ObjectId,
                "timestamp": datetime,
                "plot": {
                    "character": str,
                    "when": datetime,
                    "where": str,
                    "length": int,
                    "states": {
                        "velocity_unit_code": "kmph",
                        "thrust_unit_code": "percentage",
                        "brake_unit_code": "percentage",
                        "length": int,
                    },
                    "actions": {
                        "action_row_number": int,
                        "action_column_number": int,
                    },
                    "rewards": {
                        "reward_unit": "wh",
                    },
                },
                "history": [
                    {
                        "states": [float],  # velocity, thrust, brake
                        "actions": [float],  # pedal map of reduced_row_number
                        "action_start_row": int,
                        "reward": float,  # scalar
                    }
                ],
            }
        )

    def generate_record_schemas(self):
        # current state
        self.rec_schema.append(
            {
                "_id": ObjectId,
                "timestamp": datetime,
                "plot": {"character": str, "when": datetime, "where": str},
                "observation": [float],
            }
        )

        self.rec_schema.append(
            {
                "_id": ObjectId,
                "timestamp": datetime,
                "plot": {
                    "character": str,
                    "when": datetime,
                    "where": str,
                },
                "observation": {
                    "timestamps": datetime,
                    "state": {
                        "velocity": [float],
                        "thrust": [float],
                        "brake": [float],
                    },
                    "action": [float],
                    "reward": float,
                    "next_state": {
                        "velocity": [float],
                        "thrust": [float],
                        "brake": [float],
                    },
                },
            }
        )

        self.rec_schema.append(
            {
                "_id": ObjectId,
                "timestamp": datetime,
                "plot": {
                    "character": str,
                    "when": datetime,
                    "where": str,
                    "states": {
                        "velocity_unit_code": "kmph",
                        "thrust_unit_code": "percentage",
                        "brake_unit_code": "percentage",
                        "length": int,
                    },
                    "actions": {
                        "action_row_number": int,
                        "action_column_number": int,
                    },
                    "rewards": {
                        "reward_unit": "wh",
                    },
                },
                "observation": {
                    "timestamps": datetime,
                    "state": [float],  # [(velocity, thrust, brake)]
                    "action": [float],  # [row0, row1, row2, row3, row4]
                    "action_start_row": int,
                    "reward": float,
                    "next_state": [float],  # [(velocity, thrust, brake)]
                },
            }
        )

    def add_to_episode_pool(self, pool_size=4):
        self.logger.info("Start test_pool_deposit", extra=self.dict_logger)

        for i in range(pool_size):
            self.get_an_episode()
            self.logger.info("An Episode is created.", extra=self.dict_logger)
            self.logger.info("Start deposit an episode", extra=self.dict_logger)
            result = self.pool.deposit_item(self.episode)
            self.logger.info("Record inserted.", extra=self.dict_logger)
            self.assertEqual(result.acknowledged, True)
            pool_size = self.pool.count_items(
                vehicle_id=self.truck.vid, driver_id="zheng-longfei"
            )
            self.logger.info(f"Pool has {pool_size} records", extra=self.dict_logger)
            epi_inserted = self.pool.find_item(result.inserted_id)
            self.logger.info("episode found.", extra=self.dict_logger)
            self.assertEqual(epi_inserted["timestamp"], self.episode["timestamp"])
            self.assertEqual(epi_inserted["plot"], self.episode["plot"])
            self.assertEqual(epi_inserted["history"], self.episode["history"])

    def get_an_episode(self):
        self.logger.info("Start get_an_episode", extra=self.dict_logger)
        self.h_t = []
        # action
        k0 = 0
        N0 = 5
        map2d_5rows = self.vcu_calib_table_default[k0 : k0 + N0, :].reshape(-1).tolist()
        wh1 = 0

        action_row_number = N0
        action_column_number = self.vcu_calib_table_default.shape[1]
        action_start_row = k0
        timestamp0 = datetime.now()
        observation_length = 0
        prev_o_t = None
        prev_a_t = None

        for i in range(5):
            self.native_get()
            out = np.split(self.observation, [1, 4, 5], axis=1)  # split by empty string
            (ts, o_t0, gr_t, pow_t) = [np.squeeze(e) for e in out]
            o_t = o_t0.reshape(-1)
            ui_sum = np.sum(np.prod(pow_t, axis=1))
            wh = ui_sum / 3600.0 / self.truck.CloudSignalFrequency  # convert to Wh

            if i % 2 == 0:
                prev_r_t = wh1 + wh
                if i > 0:
                    if i == 2:
                        self.h_t = [
                            {
                                "states": prev_o_t.tolist(),
                                "actions": prev_a_t,
                                "action_start_row": action_start_row,
                                "reward": prev_r_t,
                            }
                        ]
                    else:
                        self.h_t.append(
                            {
                                "states": prev_o_t.tolist(),
                                "actions": prev_a_t,
                                "action_start_row": action_start_row,
                                "reward": prev_r_t,
                            }
                        )

                else:
                    timestamp0 = datetime.fromtimestamp(ts[0])
                    observation_length = o_t0.shape[0]
            else:
                wh1 = wh

            a_t = map2d_5rows
            prev_o_t = o_t
            prev_a_t = a_t

        self.episode = {
            "timestamp": timestamp0,
            "plot": {
                "character": self.truck.vid,
                "when": timestamp0,
                "where": "campus",
                "length": len(self.h_t),
                "states": {
                    "velocity_unit_code": "kmph",
                    "thrust_unit_code": "percentage",
                    "brake_unit_code": "percentage",
                    "length": observation_length,
                },
                "actions": {
                    "action_row_number": action_row_number,
                    "action_column_number": action_column_number,
                },
                "rewards": {
                    "reward_unit": "wh",
                },
            },
            "history": self.h_t,
        }

        self.logger.info("End get_an_episode", extra=self.dict_logger)

    def get_ddpg_record(self):
        self.ddpg_schema = {
            "_id": ObjectId,
            "timestamp": datetime,
            "plot": {
                "character": str,
                "when": datetime,
                "where": str,
                "states": {
                    "velocity_unit_code": "kmph",
                    "thrust_unit_code": "percentage",
                    "brake_unit_code": "percentage",
                    "length": int,
                },
                "actions": {
                    "action_row_number": int,
                    "action_column_number": int,
                    "action_start_row": int,
                },
                "rewards": {
                    "reward_unit": "wh",
                },
            },
            "observation": {
                "timestamps": datetime,
                "state": [float],  # [(velocity, thrust, brake)]
                "action": [float],  # [row0, row1, row2, row3, row4]
                "reward": float,
                "next_state": [float],  # [(velocity, thrust, brake)]
            },
        }
        # current state
        self.native_get()
        out = np.split(self.observation, [1, 4, 5], axis=1)  # split by empty string
        (timestamp0, motion_states0, gear_states0, power_states0) = [
            np.squeeze(e) for e in out
        ]
        ui_sum0 = (
            np.sum(np.prod(power_states0, axis=1)) / 3600.0 * 0.02
        )  # convert to Wh

        # action
        k0 = 0
        N0 = 4
        map2d_5rows = self.vcu_calib_table_default.iloc[k0 : k0 + N0]

        swap = (False,)
        self.logger.info(
            f"Create torque map: from {k0}th to the {k0+N0-1}th row.",
            extra=self.dict_logger,
        )

        # next state
        self.native_get()
        out = [
            np.squeeze(e) for e in np.split(self.observation, [1, 4, 5], axis=1)
        ]  # split by empty string
        (timestamp1, motion_states1, gear_states1, power_states1) = [
            np.squeeze(e) for e in out
        ]
        ui_sum1 = (
            np.sum(np.prod(power_states1, axis=1)) / 3600.0 * 0.02
        )  # convert to Wh

        cycle_reward = ui_sum1 + ui_sum0

        self.ddpg_record = {
            "timestamp": datetime.fromtimestamp(timestamp0[0]),
            "plot": {
                "character": self.truck.vid,
                "when": datetime.fromtimestamp(timestamp0[0]),
                "where": "campus",
                "states": {
                    "velocity_unit_code": "kmph",
                    "thrust_unit_code": "percentage",
                    "brake_unit_code": "percentage",
                    "length": motion_states0.shape[0],
                },
                "actions": {
                    "action_row_number": N0,
                    "action_column_number": self.vcu_calib_table_default.shape[1],
                    "action_start_row": k0,
                },
                "reward": {
                    "reward_unit": "wh",
                },
            },
            "observation": {
                "state": motion_states0.tolist(),
                "action": map2d_5rows.values.tolist(),
                "reward": cycle_reward,
                "next_state": motion_states1.tolist(),
            },
        }

    def get_records(self):
        # current state
        self.native_get()
        out = np.split(self.observation, [1, 4, 5], axis=1)  # split by empty string
        (timestamp0, motion_states0, gear_states0, power_states0) = [
            np.squeeze(e) for e in out
        ]
        ui_sum0 = (
            np.sum(np.prod(power_states0, axis=1)) / 3600.0 * 0.02
        )  # convert to Wh
        self.record.append(
            {
                "timestamp": datetime.fromtimestamp(timestamp0[0]),
                "plot": {
                    "character": self.truck.vid,
                    "when": datetime.fromtimestamp(timestamp0[0]),
                    "where": "campus",
                },
                "observation": self.observation.tolist(),
            }
        )

        # action
        k0 = 0
        N0 = 5
        map2d_5rows = self.vcu_calib_table_default[k0 : k0 + N0, :].reshape(-1).tolist()
        self.logger.info(
            f"Create torque map: from {k0}th to the {k0+N0-1}th row.",
            extra=self.dict_logger,
        )

        # next state
        self.native_get()
        out = [
            np.squeeze(e) for e in np.split(self.observation, [1, 4, 5], axis=1)
        ]  # split by empty string
        (timestamp1, motion_states1, gear_states1, power_states1) = [
            np.squeeze(e) for e in out
        ]
        ui_sum1 = (
            np.sum(np.prod(power_states1, axis=1)) / 3600.0 * 0.02
        )  # convert to Wh

        cycle_reward = ui_sum1 + ui_sum0

        self.record.append(
            {
                "timestamp": datetime.fromtimestamp(timestamp0[0]),
                "plot": {
                    "character": self.truck.vid,
                    "when": datetime.fromtimestamp(timestamp0[0]),
                    "where": "campus",
                    "states": {
                        "velocity_unit_code": "kmph",
                        "thrust_unit_code": "percentage",
                        "brake_unit_code": "percentage",
                        "length": motion_states0.shape[0],
                    },
                    "actions": {
                        "action_row_number": N0,
                        "action_column_number": self.vcu_calib_table_default.shape[1],
                        "action_start_row": k0,
                    },
                    "reward": {
                        "reward_unit": "wh",
                    },
                },
                "observation": {
                    "state": {
                        "velocity": motion_states0[:, 0].tolist(),
                        "thrust": motion_states0[:, 1].tolist(),
                        "brake": motion_states0[:, 2].tolist(),
                    },
                    "action": map2d_5rows,
                    "reward": cycle_reward,
                    "next_state": {
                        "velocity": motion_states1[:, 0].tolist(),
                        "thrust": motion_states1[:, 1].tolist(),
                        "brake": motion_states1[:, 2].tolist(),
                    },
                },
            }
        )
        self.record.append(
            {
                "timestamp": datetime.fromtimestamp(timestamp0[0]),
                "plot": {
                    "character": self.truck.vid,
                    "when": datetime.fromtimestamp(timestamp0[0]),
                    "where": "campus",
                    "states": {
                        "velocity_unit_code": "kmph",
                        "thrust_unit_code": "percentage",
                        "brake_unit_code": "percentage",
                        "length": motion_states0.shape[0],
                    },
                    "actions": {
                        "action_row_number": N0,
                        "action_column_number": self.vcu_calib_table_default.shape[1],
                        "action_start_row": k0,
                    },
                    "reward": {
                        "reward_unit": "wh",
                    },
                },
                "observation": {
                    "state": motion_states0.tolist(),
                    "action": map2d_5rows,
                    "reward": cycle_reward,
                    "next_state": motion_states1.tolist(),
                },
            }
        )

    # motion_states0.tolist(), map2d_5rows, cycle_reward, motion_states1.tolist())

    def add_to_record_pool(self, pool_size=4):
        for i in range(pool_size):
            self.get_ddpg_record()
            self.pool.deposit_item(self.ddpg_record)

            pool_size = self.pool.count_items(
                truck_id=self.truck.vid, driver_id="zheng-longfei"
            )
            self.logger.info(f"Pool has {pool_size} records", extra=self.dict_logger)

    def native_get(self):
        timeout = self.truck.CloudUnitNumber + 9
        signal_success, remotecan_data = self.client.get_signals(
            duration=self.truck.CloudUnitNumber, timeout=timeout
        )
        self.logger.info(
            f"get_signal(), return state:{signal_success}",
            extra=self.dict_logger,
        )

        data_type = type(remotecan_data)
        # self.logger.info(f"data type: {data_type}")
        if not isinstance(remotecan_data, dict):
            raise TypeError("udp sending wrong data type!")
        if signal_success == 0:
            try:
                # json_string = json.dumps(
                #     json_ret, indent=4, sort_keys=True, separators=(",", ": ")
                # )
                # print(f"print whole json string:{json_string}")

                self.logger.info("convert remotecan_data", extra=self.dict_logger)
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
                        timestamp = timestamps_from_can_strings(
                            can_timestamp_strings=value["list_timestamps"],
                            signal_freq=signal_freq,
                            unit_num=unit_num,
                            unit_duration=unit_duration,
                        )

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
                            timestamp.reshape((-1, 1)),
                            velocity.reshape(-1, 1),
                            thrust.reshape(-1, 1),
                            brake.reshape(-1, 1),
                            gears.reshape(-1, 1),
                            current.reshape(-1, 1),
                            voltage.reshape(-1, 1),
                        ]  # 1 + 4 + 2
                    else:
                        self.logger.info(
                            f"show status: {key}:{value}",
                            extra=self.dict_logger,
                        )
            except Exception as X:
                self.logger.error(
                    f"show status: exception {X}, data corruption",
                    extra=self.dict_logger,
                )
                return
        else:
            self.logger.error(
                f"Upload corrupt! remotecan_data: {remotecan_data}",
                extra=self.dict_logger,
            )


if __name__ == "__main__":
    unittest.main(argv=["submodule-remotecan-pool-test"], exit=False)
