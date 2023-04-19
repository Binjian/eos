from __future__ import annotations
from collections import namedtuple
import re

#  Define TypedDict for type hinting of typed collections: records and episodes

DB_CONFIG = namedtuple(
    'DB_CONFIG',
    [
        'SRVName',  # name of the server
        'DatabaseName',  # name of the database
        'RecCollName',  # name of the collection
        'EpiCollName',  # name of the collection
        'Host',  # host name for the database server
        'Port',  # port for the database server
        'Username',  # username for the database server
        'Password',  # password for the database server
        'Proxy',  # proxy for the database server
        'type',  # type of the database server: Record or Episode
    ],
)


db_config_list = [
    DB_CONFIG(
        SRVName='mongo_local',  # name of the database
        DatabaseName='eos',  # name of the database
        RecCollName='record5',  # name of the collection
        EpiCollName='episode5',  # name of the collection
        Host='127.0.0.1',  # url for the database server
        Port='27017',  # port for the database server
        Username='',  # username for the database server
        Password='',  # password for the database server
        Proxy='',  # proxy for the database server
        type='RECORD',
    ),
    DB_CONFIG(
        SRVName='mongo_ivy',  # name of the database
        DatabaseName='eos',  # name of the database
        RecCollName='record',  # name of the collection
        EpiCollName='episode',  # name of the collection
        Host='10.10.10.31',  # url for the database server
        Port='27017',  # port for the database server
        Username='',  # username for the database server
        Password='',  # password for the database server
        Proxy='',  # proxy for the database server
        type='RECORD',
    ),
    DB_CONFIG(
        SRVName='mongo_dill',  # name of the database
        DatabaseName='eos',  # name of the database
        RecCollName='record',  # name of the collection
        EpiCollName='episode',  # name of the collection
        Host='10.10.10.13',  # url for the database server
        Port='27017',  # port for the database server
        Username='',  # username for the database server
        Password='',  # password for the database server
        Proxy='',  # proxy for the database server
        type='RECORD',
    ),
    DB_CONFIG(
        SRVName='mongo_intra_sloppy',  # name of the database
        DatabaseName='eos',  # name of the database
        RecCollName='record',  # name of the collection
        EpiCollName='episode',  # name of the collection
        Host='10.0.64.64',  # url for the database server
        Port='30116',  # port for the database server
        Username='root',  # username for the database server
        Password='Newrizon123',  # password for the database server
        Proxy='',  # proxy for the database server
        type='RECORD',
    ),
    DB_CONFIG(
        SRVName='mongo_cloud',  # name of the database
        DatabaseName='eos',  # name of the database
        RecCollName='record',  # name of the collection
        EpiCollName='episode',  # name of the collection
        Host='10.10.0.7',  # url for the database server
        Port='30116',  # port for the database server
        Username='',  # username for the database server
        Password='',  # password for the database server
        Proxy='',  # proxy for the database server
        type='RECORD',
    ),
    DB_CONFIG(
        SRVName='mongo_cluster',  # name of the database
        DatabaseName='eos',  # name of the database
        RecCollName='record',  # name of the collection
        EpiCollName='episode',  # name of the collection
        Host='10.10.0.4',  # url for the database server
        Port='23000',  # port for the database server
        Username='admin',  # username for the database server
        Password='ty02ydhVqDj3QFjT',  # password for the database server
        Proxy='',  # proxy for the database server
        type='RECORD',
    ),
    DB_CONFIG(
        SRVName='mongo_cluster_intra',  # name of the database
        DatabaseName='eos',  # name of the database
        RecCollName='record',  # name of the collection
        EpiCollName='episode',  # name of the collection
        Host='10.0.48.115',  # url for the database server
        Port='23000',  # port for the database server
        Username='admin',  # username for the database server
        Password='ty02ydhVqDj3QFjT',  # password for the database server
        Proxy='',  # proxy for the database server
        type='RECORD',
    ),
    DB_CONFIG(
        SRVName='hostdb',  # name of the database, in the same bridge network of the docker host
        DatabaseName='eos',  # name of the database
        RecCollName='record',  # name of the collection
        EpiCollName='episode',  # name of the collection
        Host='hostdb',  # url for the database server
        Port='27017',  # port for the database server
        Username='',  # username for the database server
        Password='',  # password for the database server
        Proxy='',  # proxy for the database server
        type='RECORD',
    ),
]

db_config_servers_by_name = dict(
    zip([db_config.SRVName for db_config in db_config_list], db_config_list)
)
db_config_servers_by_host = dict(
    zip([db_config.Host for db_config in db_config_list], db_config_list)
)


def get_db_config(db_key: str) -> DB_CONFIG:
    """Get the db config.

    Args:
        db_key (str): string for db server name or format "usr:password@host:port"

    Returns:
        dict: db_config
    """

    # p is the validation pattern for pool_key as mongodb login string "usr:password@host:port"
    login_p = re.compile(r'^[A-Za-z]\w*:\w+@\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d{1,5}')
    assert 'mongo' in db_key or login_p.match(db_key), (
        f'Wrong format for db key {db_key}! '
        'It should be either the name of the db server (containing substring "mongo") or '
        'the format "usr:password@host:port"'
    )

    if 'mongo' in db_key.lower():
        db_config = db_config_servers_by_name.get(db_key)
        assert db_config is not None, f'No database found for db_key {db_key}!'
    else:
        # if not given as name then parse the format "usr:password@host:port"
        account_server = [s.split(':') for s in db_key.split('@')]
        flat_account_server = [s for sg in account_server for s in sg]
        assert (len(account_server) == 1 and len(flat_account_server) == 2) or (
            len(account_server) == 2 and len(flat_account_server) == 4
        ), f'Wrong format for db key {db_key}!'
        if len(account_server) == 1:
            db_config = db_config_servers_by_host.get(flat_account_server[0])
            assert (
                db_config is not None and db_config.Port == flat_account_server[1]
            ), f'Config mismatch for db key {db_key}!'

        else:
            db_config = db_config_servers_by_host.get(flat_account_server[2])
            assert (
                db_config is not None
                and db_config.Port == flat_account_server[3]
                and db_config.Username == flat_account_server[0]
                and db_config.Password == flat_account_server[1]
            ), f'Config mismatch for db server {db_key}!'

    return db_config
