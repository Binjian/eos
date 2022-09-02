from collections import namedtuple
from datetime import datetime
from bson import ObjectId

DB = namedtuple(
    "DB",
    [
        "SRVName",  # name of the server
        "DatabaseName",  # name of the database
        "CollName",  # name of the collection
        "Url",  # url for the database server
        "Port",  # port for the database server
        "Username",  # username for the database server
        "Password",  # password for the database server
        "Proxy",  # proxy for the database server
    ],
)
db_record_list = [
    DB(
        SRVName="local",  # name of the database
        DatabaseName="eos",  # name of the database
        CollName="record",  # name of the collection
        Url="127.0.0.1",  # url for the database server
        Port=27017,  # port for the database server
        Username="",  # username for the database server
        Password="",  # password for the database server
        Proxy="",  # proxy for the database server
    ),
    DB(
        SRVName="ivy",  # name of the database
        DatabaseName="eos",  # name of the database
        CollName="record",  # name of the collection
        Url="10.10.10.31",  # url for the database server
        Port=27017,  # port for the database server
        Username="",  # username for the database server
        Password="",  # password for the database server
        Proxy="",  # proxy for the database server
    ),
    DB(
        SRVName="dill",  # name of the database
        DatabaseName="eos",  # name of the database
        CollName="record",  # name of the collection
        Url="10.10.10.13",  # url for the database server
        Port=27017,  # port for the database server
        Username="",  # username for the database server
        Password="",  # password for the database server
        Proxy="",  # proxy for the database server
    ),
    DB(
        SRVName="remote_sloppy",  # name of the database
        DatabaseName="eos",  # name of the database
        CollName="record",  # name of the collection
        Url="10.0.64.64",  # url for the database server
        Port="30116",  # port for the database server
        Username="root",  # username for the database server
        Password="Newrizon123",  # password for the database server
        Proxy="",  # proxy for the database server
    ),
]
db_episode_list = [
    DB(
        SRVName="local",  # name of the database
        DatabaseName="eos",  # name of the database
        CollName="episode",  # name of the collection
        Url="127.0.0.1",  # url for the database server
        Port=27017,  # port for the database server
        Username="",  # username for the database server
        Password="",  # password for the database server
        Proxy="",  # proxy for the database server
    ),
    DB(
        SRVName="ivy",  # name of the database
        DatabaseName="eos",  # name of the database
        CollName="episode",  # name of the collection
        Url="10.10.10.31",  # url for the database server
        Port=27017,  # port for the database server
        Username="",  # username for the database server
        Password="",  # password for the database server
        Proxy="",  # proxy for the database server
    ),
    DB(
        SRVName="dill",  # name of the database
        DatabaseName="eos",  # name of the database
        CollName="episode",  # name of the collection
        Url="10.10.10.13",  # url for the database server
        Port=27017,  # port for the database server
        Username="",  # username for the database server
        Password="",  # password for the database server
        Proxy="",  # proxy for the database server
    ),
    DB(
        SRVName="remote_sloppy",  # name of the database
        DatabaseName="eos",  # name of the database
        CollName="episode",  # name of the collection
        Url="10.0.64.64",  # url for the database server
        Port="30116",  # port for the database server
        Username="root",  # username for the database server
        Password="Newrizon123",  # password for the database server
        Proxy="",  # proxy for the database server
    ),
]

dbs_record = dict(zip([db.SRVName for db in db_record_list], db_record_list))
dbs_episode = dict(zip([db.SRVName for db in db_episode_list], db_episode_list))

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
            "plot": {"character": str, "when": datetime, "where": str},
            "observation": [float],
        },  # structure of the schema
    ),
    SCHEMA(
        NAME="record_deep",  # name of the schema
        STRUCTURE={
            "_id": ObjectId,
            "timestamp": datetime,
            "plot": {
                "character": str,
                "when": datetime,
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
        },  # structure of the schema
    ),
    SCHEMA(
        NAME="record_deeper",  # name of the schema
        STRUCTURE={
            "_id": ObjectId,
            "timestamp": datetime,
            "plot": {
                "character": str,
                "when": datetime,
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
                    "action_start_row": int,
                },
                "rewards": {
                    "reward_unit": "wh",
                },
            },
            "observation": {
                "timestamps": datetime,
                "state": {"velocity": [float], "thrust": [float], "brake": [float]},
                "action": [float],
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
            "timestamp": datetime,
            "plot": {
                "character": str,
                "when": datetime,
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
            "timestamp": datetime,
            "plot": {
                "character": str,
                "when": datetime,
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
                "rewards": {
                    "reward_unit": "wh",
                },
            },
            "history": [
                {
                    "states": [float],  # velocity, thrust, brake
                    "actions": [float],  # pedal map of reduced_row_number
                    "reward": float,  # scalar
                }
            ],
        },
    ),
]

episode_schemas = dict(
    zip([schema.NAME for schema in epi_schema_list], epi_schema_list)
)
