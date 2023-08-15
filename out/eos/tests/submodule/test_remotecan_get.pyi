import unittest

from _typeshed import Incomplete

from eos import RemoteCanClient as RemoteCanClient
from eos import projroot as projroot
from eos.config import can_servers as can_servers
from eos.config import generate_vcu_calibration as generate_vcu_calibration
from eos.config import trip_servers as trip_servers
from eos.config import trucks_by_id as trucks_by_id
from eos.config import trucks_by_vin as trucks_by_vin
from eos.utils import ragged_nparray_list_interp as ragged_nparray_list_interp

class TestRemoteCanGet(unittest.TestCase):
    site: str
    proxies: Incomplete
    proxies_socks: Incomplete
    proxies_lantern: Incomplete
    trucks: Incomplete
    truck_name: str
    projroot: Incomplete
    logger: Incomplete
    dictLogger: Incomplete
    truck: Incomplete
    vcu_calib_table_default: Incomplete
    def setUp(self) -> None: ...
    def set_logger(self, projroot) -> None: ...
    def native_get(self) -> None: ...
    def native_send(self) -> None: ...
