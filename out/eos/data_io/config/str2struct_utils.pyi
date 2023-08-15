from typing import Union

from eos.data_io.config import RE_DBKEY as RE_DBKEY
from eos.data_io.config import RE_DRIVER as RE_DRIVER
from eos.data_io.config import RE_VIN as RE_VIN
from eos.data_io.config import CANMessenger as CANMessenger
from eos.data_io.config import DBConfig as DBConfig
from eos.data_io.config import Driver as Driver
from eos.data_io.config import TripMessenger as TripMessenger
from eos.data_io.config import TruckInCloud as TruckInCloud
from eos.data_io.config import TruckInField as TruckInField
from eos.data_io.config import can_servers_by_host as can_servers_by_host
from eos.data_io.config import can_servers_by_name as can_servers_by_name
from eos.data_io.config import drivers_by_id as drivers_by_id
from eos.data_io.config import trip_servers_by_host as trip_servers_by_host
from eos.data_io.config import trip_servers_by_name as trip_servers_by_name
from eos.data_io.config import trucks_by_id as trucks_by_id
from eos.data_io.config import trucks_by_vin as trucks_by_vin

def str_to_truck(truck_str: str) -> Union[TruckInCloud, TruckInField]: ...
def str_to_driver(driver_str: str) -> Driver: ...
def str_to_can_server(can_server_str: str) -> CANMessenger: ...
def str_to_trip_server(trip_server_str: str) -> TripMessenger: ...
