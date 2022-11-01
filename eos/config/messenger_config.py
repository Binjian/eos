from collections import namedtuple
from datetime import datetime

from bson import ObjectId

CANMessenger = namedtuple(
    "CANMessenger",
    [
        "SRVName",  # name of the server
        "Url",  # url for the database server
        "Port",  # port for the database server
    ],
)
TripMessenger = namedtuple(
    "TripMessenger",
    [
        "SRVName",  # name of the server
        "Url",  # url for the database server
        "Port",  # port for the database server
    ],
)
canserver_list = [
    CANMessenger(
        SRVName="newrizon_test",  # name of the database
        Url="10.0.64.78",  # url for the database server
        Port="5000",  # port for the database server
    ),
    CANMessenger(
        SRVName="baiduyun_k8s",  # name of the database
        Url="aidriver.veos.srv",  # url for the database server
        Port="5001",  # port for the database server
    ),
]

can_servers = dict(zip([srv.SRVName for srv in canserver_list], canserver_list))

tripserver_list = [
    TripMessenger(
        SRVName="newrizon_test",  # name of the database
        Url="10.0.64.78",  # url for the database server
        Port="9876",  # port for the database server
    ),
    TripMessenger(
        SRVName="baiduyun_k8s",  # name of the database
        Url="remotecan.veos.srv",  # url for the database server
        Port="5000",  # port for the database server
    ),
]
trip_servers = dict(zip([srv.SRVName for srv in tripserver_list], tripserver_list))