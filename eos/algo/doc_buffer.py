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
from eos.struct import DocItemT
from eos import DBPool, dictLogger, logger


@dataclass
class DocBuffer(Buffer, Generic[DocItemT]):
    """
    A Buffer connected with a database pool
    Args:
        key: is a key for mongodb (str) or DB_CONFIG (dict)
        the key leads to a config with db_name and
        collection name with a switch for record or episode:
            - string for db server name
            - or string of the format "usr:password@host:port"
                for mongo_cluster:
                    Host="10.10.0.4",  # url for the database server
                    Port="23000",  # port for the database server
                    Username="admin",  # username for the database server
                    Password="ty02ydhVqDj3QFjT",  # password for the database server
                    ==> mongo_key = "admin:ty02ydhVqDj3QFjT@10.10.0.4:23000"
    """

    db_config: DB_CONFIG
    batch_size: int = (4,)
    pool: DBPool[DocItemT] = (None,)
    query: dict = (None,)
    buffer_count: int = (0,)

    def __post_init__(self):
        self.logger = logger.getChild('main').getChild('DBBuffer')
        self.logger.propagate = True
        super().__post_init__()
        self.load()

    def load(self):

        self.query = {
            'vehicle_id': self.plot.plot_dict['character'],
            'driver_id': self.plot.plot_dict['driver'],
            'dt_start': None,
            'dt_end': None,
        }
        self.pool = DBPool[DocItemT](
            db_config=self.db_config,
            query=self.query,
        )
        self.buffer_count = self.pool.count()
        # check plot with input vehicle and driver
        batch_1 = self.pool.sample(size=1, query=self.query)
        assert self.plot.are_same_plots(
            batch_1[0].plot
        ), f'plot in db is {batch_1[0].plot}, but plot in config is {self.plot}'
        (
            num_states,
            num_actions,
        ) = self.plot.get_number_of_states_actions()  # the realtime number
        self.logger.info(
            f'Connected to MongoDB {self.db_config.DatabaseName}, '
            f'collection {self.db_config.RecCollName}, '
            f'record number {self.buffer_count}',
            f'num_states: {num_states}, num_actions: {num_actions}',
            extra=dictLogger,
        )

    def close(self):
        # self.pool.close()  # close the connection is done by pool finalizer
        pass

    def store(self, item: DocItemT):
        result = self.pool.store(item)
        return result

    def find(self, idx):
        return self.pool.find(idx)

    def sample(self) -> list[DocItemT]:
        batch = self.pool.sample(size=self.batch_size)
        return batch

    def count(self, query: Optional[dict] = None):
        self.buffer_count = self.pool.count(query)
