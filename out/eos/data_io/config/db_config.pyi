from typing import NamedTuple

from _typeshed import Incomplete

RE_DBKEY: str

class DBConfig(NamedTuple):
    SRVName: Incomplete
    DatabaseName: Incomplete
    CollName: Incomplete
    Host: Incomplete
    Port: Incomplete
    Username: Incomplete
    Password: Incomplete
    Proxy: Incomplete
    type: Incomplete

db_config_list: Incomplete
db_config_servers_by_name: Incomplete
db_config_servers_by_host: Incomplete

def get_db_config(db_key: str, coll_type: str) -> DBConfig: ...
