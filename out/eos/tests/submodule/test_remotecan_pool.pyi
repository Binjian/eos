import unittest

from _typeshed import Incomplete

from eos import Pool as Pool
from eos import RemoteCanClient as RemoteCanClient
from eos import projroot as projroot
from eos.config import can_servers_by_name as can_servers_by_name
from eos.config import db_config_servers_by_name as db_config_servers_by_name
from eos.config import episode_schemas as episode_schemas
from eos.config import generate_vcu_calibration as generate_vcu_calibration
from eos.config import record_schemas as record_schemas
from eos.config import trucks_by_id as trucks_by_id
from eos.config import trucks_by_vin as trucks_by_vin
from eos.utils import ragged_nparray_list_interp as ragged_nparray_list_interp
from eos.utils.exception import TruckIDError as TruckIDError

class TestRemoteCanPool(unittest.TestCase):
    site: str
    proxies: Incomplete
    proxies_socks: Incomplete
    proxies_lantern: Incomplete
    trucks_by_id: Incomplete
    truck_name: str
    db_server_name: str
    db_server: Incomplete
    can_server_name: str
    can_server: Incomplete
    record_schemas: Incomplete
    episode_schemas: Incomplete
    rec_schema: Incomplete
    epi_schema: Incomplete
    record: Incomplete
    projroot: Incomplete
    logger: Incomplete
    dictLogger: Incomplete
    truck: Incomplete
    observe_length: Incomplete
    vcu_calib_table_default: Incomplete
    def setUp(self) -> None: ...
    def set_logger(self, projroot) -> None: ...
    client: Incomplete
    epi_sch: Incomplete
    pool: Incomplete
    def test_native_pool_deposit_episode(self) -> None: ...
    def test_native_pool_sample_episode(self) -> None: ...
    rec_sch: Incomplete
    def test_native_pool_sample_record(self) -> None: ...
    def test_native_pool_deposit_record(self) -> None: ...
    def test_native_pool_consecutive_observations(self) -> None: ...
    def test_native_pool_consecutive_records(self) -> None: ...
    def test_native_pool_consecutive_flash_test(self) -> None: ...
    def native_send(self) -> None: ...
    def generate_epi_schemas(self) -> None: ...
    def generate_record_schemas(self) -> None: ...
    def add_to_episode_pool(self, pool_size: int = ...) -> None: ...
    h_t: Incomplete
    episode: Incomplete
    def get_an_episode(self) -> None: ...
    ddpg_schema: Incomplete
    ddpg_record: Incomplete
    def get_ddpg_record(self) -> None: ...
    def get_records(self) -> None: ...
    def add_to_record_pool(self, pool_size: int = ...) -> None: ...
    observation: Incomplete
    def native_get(self) -> None: ...
