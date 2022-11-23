from collections import namedtuple
from datetime import datetime

from bson import ObjectId

CANMessenger = namedtuple(
    "CANMessenger",
    [
        "SRVName",  # name of the server
        "Host",  # url for the database server
        "Port",  # port for the database server
    ],
)
TripMessenger = namedtuple(
    "TripMessenger",
    [
        "SRVName",  # name of the server
        "Host",  # url for the database server
        "Port",  # port for the database server
    ],
)
canserver_list = [
    CANMessenger(
        SRVName="newrizon_test",  # name of the database
        Host="10.0.64.78",  # url for the database server
        Port="5000",  # port for the database server
    ),
    CANMessenger(
        SRVName="baiduyun_k8s",  # name of the database
        Host="10.10.0.7",  # url for the database server
        Port="30865",  # port for the database server
    ),
]

can_servers_by_name = dict(zip([srv.SRVName for srv in canserver_list], canserver_list))
can_servers_by_host = dict(zip([srv.Host for srv in canserver_list], canserver_list))

tripserver_list = [
    TripMessenger(
        SRVName="newrizon_test",  # name of the database
        Host="10.0.64.78",  # url for the database server
        Port="9876",  # port for the database server
    ),
    TripMessenger(
        SRVName="baiduyun_k8s",  # name of the database
        Host="10.10.0.13",  # url for the database server
        Port="9876",  # port for the database server
    ),
]
trip_servers_by_name = dict(
    zip([srv.SRVName for srv in tripserver_list], tripserver_list)
)
trip_servers_by_host = dict(zip([srv.Host for srv in tripserver_list], tripserver_list))
