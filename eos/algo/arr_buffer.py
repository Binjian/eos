from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from configparser import ConfigParser

from .buffer import Buffer
from eos.struct import ArrItemT
from eos import RecordFilePool, EpisodeFilePool, dictLogger, logger


@dataclass
class ArrBuffer(Buffer[ArrItemT]):
    """
    A Buffer connected with a data array file pool
    Args:
        data_folder: is a folder for the data files and the config file
        config_file: is a filename for the config file
    """

    recipe: ConfigParser
    batch_size: int = (4,)
    pool: [RecordFilePool | EpisodeFilePool] = None
    query: dict = (None,)
    buffer_count: int = (0,)

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
        self.query = {
            'vehicle_id': self.plot.plot_dict['character'],
            'driver_id': self.plot.plot_dict['driver'],
            'dt_start': None,
            'dt_end': None,
        }

        # connect to the array pool
        if 'RECORD' in self.coll_type.upper():
            self.pool = RecordFilePool(
                recipe=self.recipe,
                query=self.query,
            )
        elif 'EPISODE' in self.coll_type.upper():
            self.pool = EpisodeFilePool(
                recipe=self.recipe,
                query=self.query,
            )
        else:
            raise ValueError(f'Unknown buffer type: {self._name}')

        self.buffer_count = self.pool.count()

        number_states, number_actions = self.plot.get_number_of_states_actions()
        self.logger.info(
            f'Connected to RecordFilePool {self.recipe["DEFAULT"]["data_folder"]}, '
            f'record number {self.buffer_count}',
            f'num_states: {number_states}, num_actions: {number_actions}',
            extra=dictLogger,
        )

    def close(self):
        # self.pool.close()  # close the connection is done by pool finalizer
        pass

    def store(self, item: ArrItemT):
        self.pool.store(item)

    def find(self, idx):
        return self.pool.find(idx)

    def sample(self) -> list[ArrItemT]:
        batch = self.pool.sample(size=self.batch_size)
        return batch

    def count(self, query: Optional[dict] = None):
        self.buffer_count = self.pool.count(query)
