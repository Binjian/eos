from collections import namedtuple
from datetime import datetime
from typing import TypedDict, NotRequired, Any
from bson import ObjectId
from numpy import ndarray

DB_CONFIG = namedtuple(
    "DB_CONFIG",
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
db_config_list = [
    DB_CONFIG(
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
    DB_CONFIG(
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
    DB_CONFIG(
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
    DB_CONFIG(
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
    DB_CONFIG(
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
    DB_CONFIG(
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
    DB_CONFIG(
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
    DB_CONFIG(
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

db_config_servers_by_name = dict(
    zip([db_config.SRVName for db_config in db_config_list], db_config_list)
)
db_config_servers_by_host = dict(
    zip([db_config.Host for db_config in db_config_list], db_config_list)
)

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


def get_db_config(db_key: str) -> DB_CONFIG:
    """Get the db config.

    Args:
        db_key (str): string for db server name or format "usr:password@host:port"

    Returns:
        dict: db_config
    """

    db_config = db_config_servers_by_name.get(db_key)
    if (
        db_config is None
    ):  # if not given as name then parse the format "usr:password@host:port"
        account_server = [s.split(":") for s in db_key.split("@")]
        flat_account_server = [s for sg in account_server for s in sg]
        assert (len(account_server) == 1 and len(flat_account_server) == 2) or (
            len(account_server) == 2 and len(flat_account_server) == 4
        ), f"Wrong format for db key {db_key}!"
        if len(account_server) == 1:
            db_config = db_config_servers_by_host.get(flat_account_server[0])
            assert (
                db_config is not None and db_config.Port == flat_account_server[1]
            ), f"Config mismatch for db key {db_key}!"

        else:
            db_config = db_config_servers_by_host.get(flat_account_server[2])
            assert (
                db_config is not None
                and db_config.Port == flat_account_server[3]
                and db_config.Username == flat_account_server[0]
                and db_config.Password == flat_account_server[1]
            ), f"Config mismatch for db server {db_key}!"

    return db_config


#  Define TypedDict for type hinting of typed collections: records and episodes


class StateSpecs(TypedDict):
    """Observation of the episode."""

    observation_specs: list[dict[str, str]]
    unit_number: int
    unit_duration: int
    frequency: int


class Plot(TypedDict):
    """Plot of the item specs"""

    character: str
    driver: str
    when: datetime
    tz: NotRequired[str]
    where: str
    length: NotRequired[int]  # only for episode property, not for record
    state_specs: StateSpecs
    action_specs: dict[str, int]
    reward_specs: dict[str, str]


class Record(TypedDict):
    """Record of the observation"""

    _id: NotRequired[ObjectId]  # for record, _id is generated by pymongo, not required
    timestamp: datetime
    plot: Plot
    observation: dict[str, list[dict[str, list[Any]]]]

class RecordPlain(TypedDict):
    """Record of the observation"""
    episode_starts: datetime
    timestamps: ndarray
    states: ndarray
    actions: ndarray
    rewards: float
    next_states: ndarray
    table_start_rows: int

class Episode(TypedDict):
    """Episode of the record"""

    _id: NotRequired[ObjectId]  # for record, _id is generated by pymongo, not required
    timestamp: datetime
    plot: Plot
    history: list[dict[str, list[Any]]]

class EpisodePlain(TypedDict):
    """Episode of the record"""
    episode_starts: datetime
    states: ndarray
    actions: ndarray
    action_start_rows: ndarray
    rewards: float