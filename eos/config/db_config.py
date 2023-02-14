from collections import namedtuple
from datetime import datetime

from bson import ObjectId

DB = namedtuple(
    "DB",
    [
        "SRVName",  # name of the server
        "DatabaseName",  # name of the database
        "RecCollName",  # name of the collection
        "EpiCollName",  # name of the collection
        "Host",  # host name for the database server
        "Port",  # port for the database server
        "Username",  # username for the database server
        "Password",  # password for the database server
        "Proxy",  # proxy for the database server
    ],
)
db_list = [
    DB(
        SRVName="mongo_local",  # name of the database
        DatabaseName="eos",  # name of the database
        RecCollName="record5",  # name of the collection
        EpiCollName="episode5",  # name of the collection
        Host="127.0.0.1",  # url for the database server
        Port="27017",  # port for the database server
        Username="",  # username for the database server
        Password="",  # password for the database server
        Proxy="",  # proxy for the database server
    ),
    DB(
        SRVName="mongo_ivy",  # name of the database
        DatabaseName="eos",  # name of the database
        RecCollName="record",  # name of the collection
        EpiCollName="episode",  # name of the collection
        Host="10.10.10.31",  # url for the database server
        Port="27017",  # port for the database server
        Username="",  # username for the database server
        Password="",  # password for the database server
        Proxy="",  # proxy for the database server
    ),
    DB(
        SRVName="mongo_dill",  # name of the database
        DatabaseName="eos",  # name of the database
        RecCollName="record",  # name of the collection
        EpiCollName="episode",  # name of the collection
        Host="10.10.10.13",  # url for the database server
        Port="27017",  # port for the database server
        Username="",  # username for the database server
        Password="",  # password for the database server
        Proxy="",  # proxy for the database server
    ),
    DB(
        SRVName="mongo_intra_sloppy",  # name of the database
        DatabaseName="eos",  # name of the database
        RecCollName="record",  # name of the collection
        EpiCollName="episode",  # name of the collection
        Host="10.0.64.64",  # url for the database server
        Port="30116",  # port for the database server
        Username="root",  # username for the database server
        Password="Newrizon123",  # password for the database server
        Proxy="",  # proxy for the database server
    ),
    DB(
        SRVName="mongo_cloud",  # name of the database
        DatabaseName="eos",  # name of the database
        RecCollName="record",  # name of the collection
        EpiCollName="episode",  # name of the collection
        Host="10.10.0.7",  # url for the database server
        Port="30116",  # port for the database server
        Username="",  # username for the database server
        Password="",  # password for the database server
        Proxy="",  # proxy for the database server
    ),
    DB(
        SRVName="mongo_cluster",  # name of the database
        DatabaseName="eos",  # name of the database
        RecCollName="record",  # name of the collection
        EpiCollName="episode",  # name of the collection
        Host="10.10.0.4",  # url for the database server
        Port="23000",  # port for the database server
        Username="admin",  # username for the database server
        Password="ty02ydhVqDj3QFjT",  # password for the database server
        Proxy="",  # proxy for the database server
    ),
    DB(
        SRVName="mongo_cluster_intra",  # name of the database
        DatabaseName="eos",  # name of the database
        RecCollName="record",  # name of the collection
        EpiCollName="episode",  # name of the collection
        Host="10.0.48.115",  # url for the database server
        Port="23000",  # port for the database server
        Username="admin",  # username for the database server
        Password="ty02ydhVqDj3QFjT",  # password for the database server
        Proxy="",  # proxy for the database server
    ),
    DB(
        SRVName="hostdb",  # name of the database, in the same bridge network of the docker host
        DatabaseName="eos",  # name of the database
        RecCollName="record",  # name of the collection
        EpiCollName="episode",  # name of the collection
        Host="hostdb",  # url for the database server
        Port="27017",  # port for the database server
        Username="",  # username for the database server
        Password="",  # password for the database server
        Proxy="",  # proxy for the database server
    ),
]

db_servers_by_name = dict(zip([db.SRVName for db in db_list], db_list))
db_servers_by_host = dict(zip([db.Host for db in db_list], db_list))

SCHEMA = namedtuple(
    "SCHEMA",
    [
        "NAME",  # name of the schema
        "STRUCTURE",  # structure of the schema
    ],
)

rec_schema_list = [
    SCHEMA(
        NAME="record_flat",  # name of the schema
        STRUCTURE={
            "_id": ObjectId,
            "timestamp": datetime,
            "plot": {"character": str, "when": datetime, "where": str, "driver": str},
            "observation": [float],
        },  # structure of the schema
    ),
    SCHEMA(
        NAME="record_deep",  # name of the schema
        STRUCTURE={
            "_id": ObjectId,
            "timestamp": datetime,  # record datetime,when the record is created
            "plot": {
                "character": str,
                "driver": str,
                "when": datetime,  # episode datetime with tzinfo, when the episode is created
                "where": str,
                "states": {
                    "velocity_unit": "kmph",
                    "thrust_unit": "percentage",
                    "brake_unit": "percentage",
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
        },  # structure of the schema
    ),
    SCHEMA(
        NAME="record_deeper",  # name of the schema
        STRUCTURE={
            "_id": ObjectId,
            "timestamp": datetime,  # record datetime,when the record is created
            "plot": {
                "character": str,
                "driver": str,
                "when": datetime,  # episode datetime with tzinfo, when the episode is created
                "where": str,
                "states": {
                    "velocity_unit": "kmph",
                    "thrust_unit": "percentage",
                    "brake_unit": "percentage",
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
                "state": {"velocity": [float], "thrust": [float], "brake": [float]},
                "action": [float],
                "action_start_row": int,
                "reward": float,
                "next_state": {
                    "velocity": [float],
                    "thrust": [float],
                    "brake": [float],
                },
            },
        },  # structure of the schema
    ),
]
record_schemas = dict(zip([schema.NAME for schema in rec_schema_list], rec_schema_list))

epi_schema_list = [
    SCHEMA(
        NAME="episode_flat",  # name of the schema
        STRUCTURE={
            "_id": ObjectId,
            "timestamp": datetime,  # episode datetime,when the episode is created
            "plot": {
                "character": str,
                "driver": str,
                "when": datetime,  # episode datetime with tzinfo, when the episode is created
                "where": str,
                "length": int,
                "states": {
                    "velocity_unit": "kmph",
                    "thrust_unit": "percentage",
                    "brake_unit": "percentage",
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
        },
    ),
    SCHEMA(
        NAME="episode_deep",  # name of the schema
        STRUCTURE={
            "_id": ObjectId,
            "timestamp": datetime,  # episode datetime,when the episode is created
            "plot": {
                "character": str,
                "driver": str,
                "when": datetime,  # episode datetime with tzinfo, when the episode is created
                "where": str,
                "length": int,
                "states": {
                    "velocity_unit": "kmph",
                    "thrust_unit": "percentage",
                    "brake_unit": "percentage",
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
                    "action_start_row": int,  # scalar
                    "reward": float,  # scalar
                }
            ],
        },
    ),
]

episode_schemas = dict(
    zip([schema.NAME for schema in epi_schema_list], epi_schema_list)
)
