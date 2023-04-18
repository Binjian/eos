from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Generic, TypeVar, Any, get_args, Dict
from configparser import ConfigParser

from .buffer import Buffer
from eos.config import (
    Truck,
    trucks_by_name,
)
from eos.struct import ArrItemT, Plot, get_number_of_states_actions
from eos import RecordFilePool, EpisodeFilePool, dictLogger, logger

@dataclass
class ArrBuffer(Buffer, Generic[ArrItemT]):
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

    data_folder: Optional[str] = None
    config_file: Optional[str] = None
    batch_size: int = (4,)
    recipe: Optional[ConfigParser | Dict] = None
    pool: [RecordFilePool | EpisodeFilePool] = None
    query: dict = (None,)

    def __post_init__(self):
        self.logger = logger.getChild('main').getChild('DBBuffer')
        self.logger.propagate = True
        super().__post_init__()
        self.load()

    def load(self):

        # check if datafolder exists and have valid ini file for recipe
        # if not, create a new one
        # if yes, load the recipe and compare with the realtime truck signal specs
        # if not matching, raise error
        # if matching, continue

        number_states, number_actions = get_number_of_states_actions(self.plot)
        default_recipe_from_truck: ConfigParser = ConfigParser()
        default_recipe_from_truck.read_dict(
            {
                'DEFAULT': {
                    'data_folder': '.',
                    'recipe_file_name': 'recipe.ini',
                    'capacity': '300000',
                    'index': '0',
                    'full': 'False',  # flag True if the storage is full
                },
                'array_specs': {
                    'character': self.plot['character'],  # vehicle the agent
                    'driver': self.plot['driver'],
                    'states': str(number_states),  # 50*4*3
                    'actions': str(number_actions),  # 17*4
                    'rewards': '1',
                    'next_states': str(number_states),  # 50*4*3
                    'table_start_rows': '1',
                },
            }
        )
        if self.recipe is None:
            self.logger.info('No recipe specified, using default recipe')

            self.recipe = ConfigParser()
            self.recipe.read_dict(default_recipe_from_truck)
        else:
            recipe_from_file = ConfigParser()
            try:
                recipe_from_file.read(self.data_folder + '/' + self.config_file)
                # check if the recipe is matching with the truck specs
                assert (
                    recipe_from_file['array_specs']
                    == default_recipe_from_truck['array_specs']
                ), f"ini file array_specs is not matching the realtime truck signal specs"
                self.logger.info(
                    f"ini file array_specs is matching the realtime truck signal specs.",
                    extra=dictLogger,
                )
                self.logger.info(
                    f"Apply ini file DEFAULT: {recipe_from_file['DEFAULT']} ...",
                    extra=dictLogger,
                )
                self.logger.info(
                    f"... vs. default config DEFAULT: {default_recipe_from_truck['DEFAULT']}",
                    extra=dictLogger,
                )

            except FileNotFoundError:
                recipe_from_file = default_recipe_from_truck
                self.logger.info(f"ini file not found, using default recipe!")
            except Exception as e:
                self.logger.error(f'Error reading recipe file: {e}', extra=dictLogger)
                raise e

        self.query = {
            'vehicle_id': self.truck.TruckName,
            'driver_id': self.driver,
            'dt_start': None,
            'dt_end': None,
        }

        # connect to the array pool
        if 'RecordArr' in self._name:
            self.pool = RecordFilePool(
                key=self.db_config,
                query=self.query,
            )
        elif 'EpisodeArr' in self._name:
            self.pool = EpisodeFilePool(
                key=self.db_config,
                query=self.query,
            )
        else:
            raise ValueError(f'Unknown buffer type: {self._name}')

        self.buffer_count = self.pool.count()
        # check plot with input vehicle and driver
        batch_1 = self.pool.sample(size=1, query=self.query)
        print(f'batch_1: {batch_1}')
        num_states, num_actions = get_number_of_states_actions(self.plot)

        # TODO: check whether plot of the pool is matching with the truck specs
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

    def store(self, item: ArrItemT):
        result = self.pool.store(item)
        return result

    def find(self, idx):
        return self.pool.find(idx)

    def sample(self) -> list[ArrItemT]:
        batch = self.pool.sample(size=self.batch_size)
        return batch

    def count(self, query: Optional[dict] = None):
        self.buffer_count = self.pool.count(query)
