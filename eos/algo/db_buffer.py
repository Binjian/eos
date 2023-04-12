from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Generic, TypeVar, Any, get_args


from .buffer import Buffer
from eos.config import (
    DB_CONFIG,
    Truck,
    trucks_by_name,
    get_db_config,
)
from eos.struct import DBItemT
from eos import DBPool, dictLogger, logger
from .dpg import get_algo_data_info


@dataclass
class DBBuffer(Buffer, Generic[DBItemT]):
    """
    A Buffer connected with a database pool
    Args:
        key: is a key for mongodb (str) or DB_CONFIG (dict)
        the key leads to a config with db_name and
        collection name with a swtich for record or episode:
            - string for db server name
            - or string of the format "usr:password@host:port"
                for mongo_cluster:
                    Host="10.10.0.4",  # url for the database server
                    Port="23000",  # port for the database server
                    Username="admin",  # username for the database server
                    Password="ty02ydhVqDj3QFjT",  # password for the database server
                    ==> mongo_key = "admin:ty02ydhVqDj3QFjT@10.10.0.4:23000"
    """

    key: str | DB_CONFIG  # required  # if None
    truck: Truck = trucks_by_name['VB7']
    driver: str = ('longfei-zheng',)
    batch_size: int = (4,)
    padding_value: float = (0,)
    data_folder: str = ('./',)
    db_config: DB_CONFIG = (None,)
    num_states: int = (600,)
    num_actions: int = (68,)
    buffer_capacity: int = (10000,)
    buffer_count: int = (0,)
    pool: DBPool[DBItemT] = (None,)
    query: dict = (None,)

    def __post_init__(self):
        self.logger = logger.getChild('main').getChild('DBBuffer')
        self.logger.propagate = True
        super().__post_init__()
        self.load()

    def load(self):
        if isinstance(self.key, str):
            self.db_config = get_db_config(self.key)
        else:
            self.db_config = self.key

        url = (
            self.db_config.Username
            + ':'
            + self.db_config.Password
            + '@'
            + self.db_config.Host
            + ':'
            + self.db_config.Port
        )
        self.query = {
            'vehicle_id': self.truck.TruckName,
            'driver_id': self.driver,
            'dt_start': None,
            'dt_end': None,
        }
        self.pool = DBPool[DBItemT](
            key=self.key,
            query=self.query,
        )
        self.buffer_count = self.pool.count()
        # check plot with input vehicle and driver
        batch_1 = self.pool.sample(size=1, query=self.query)
        self.num_states, self.num_actions = get_algo_data_info(
            batch_1[0], self.truck, self.driver
        )
        self.logger.info(
            f'Connected to MongoDB {self.db_config.DatabaseName}, '
            f'collection {self.db_config.RecCollName}, '
            f'record number {self.buffer_count}',
            f'num_states: {self.num_states}, num_actions: {self.num_actions}',
            extra=dictLogger,
        )

    def close(self):
        # self.pool.close()  # close the connection is done by pool finalizer
        pass

    def store(self, item: DBItemT):
        result = self.pool.store(item)
        return result

    def find(self, idx):
        return self.pool.find(idx)

    def sample(self) -> list[DBItemT]:
        batch = self.pool.sample(size=self.batch_size)
        return batch

    def count(self, query: Optional[dict] = None):
        self.buffer_count = self.pool.count(query)
