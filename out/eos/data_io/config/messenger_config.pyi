from datetime import datetime as datetime
from typing import NamedTuple

from _typeshed import Incomplete
from bson import ObjectId as ObjectId

class CANMessenger(NamedTuple):
    SRVName: Incomplete
    Host: Incomplete
    Port: Incomplete

class TripMessenger(NamedTuple):
    SRVName: Incomplete
    Host: Incomplete
    Port: Incomplete

can_server_list: Incomplete
can_servers_by_name: Incomplete
can_servers_by_host: Incomplete
tripserver_list: Incomplete
trip_servers_by_name: Incomplete
trip_servers_by_host: Incomplete
