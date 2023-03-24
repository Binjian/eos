import abc
from dataclasses import dataclass
from datetime import datetime

from eos import dictLogger, logger
from eos.config import (
    DB_CONFIG,
    Truck,
    trucks_by_name,
    Plot
)


def get_algo_data_info(item: dict, truck: Truck, driver: str) -> tuple:
    """Check if the data is valid for the algorithm.

    Args:
        item (dict): data item
        truck (Truck): truck object
        driver (str): driver name

    Returns:
        bool: True if the data is valid
    """

    obs = item['plot']['states']['observations']
    assert (
        len(obs) == truck.ObservationNumber
    ), f'observation number mismatch, {len(obs)} != {truck.ObservationNumber}!'
    unit_number = item['plot']['states']['unit_number']
    assert (
        unit_number == truck.CloudUnitNumber
    ), f'unit number mismatch, {unit_number} != {truck.CloudUnitNumber}!'
    unit_duration = item['plot']['states']['unit_duration']
    assert (
        unit_duration == truck.CloudUnitDuration
    ), f'unit duration mismatch, {unit_duration} != {truck.CloudUnitDuration}!'
    frequency = item['plot']['states']['frequency']
    assert (
        frequency == truck.CloudSignalFrequency
    ), f'frequency mismatch, {frequency} != {truck.CloudSignalFrequency}!'

    action_row_number = item['plot']['actions']['action_row_number']
    assert (
        action_row_number == truck.ActionFlashRow
    ), f'action row number mismatch, {action_row_number} != {truck.ActionFlashRow}!'
    action_column_number = item['plot']['actions']['action_column_number']
    assert (
        action_column_number == truck.PedalScale
    ), f'action column number mismatch, {action_column_number} != {truck.PedalScale}!'
    truckname_in_data = item['plot']['character']
    assert (
        truckname_in_data == truck.TruckName
    ), f'truck name mismatch, {truckname_in_data} != {truck.TruckName}!'

    num_states = len(obs) * unit_number * unit_duration * frequency
    num_actions = action_row_number * action_column_number

    driver_in_data = item['plot']['driver']
    assert (
        driver_in_data == driver
    ), f'driver name mismatch, {driver_in_data} != {driver}!'

    return num_states, num_actions


"""Base class for differentiable policy gradient methods."""


@dataclass
class DPG(abc.ABC):
    _truck: Truck = (trucks_by_name['VB7'],)
    _driver: str = ('longfei',)
    _num_states: int = (600,)
    _num_actions: int = (68,)
    _buffer_capacity: int = (10000,)
    _batch_size: int = (4,)
    _hidden_unitsAC: tuple = ((256, 16, 32),)
    _action_bias: float = (0.0,)
    _n_layersAC: tuple = ((2, 2),)
    _padding_value: float = (0,)
    _gamma: float = (0.99,)
    _tauAC: tuple = ((0.005, 0.005),)
    _lrAC: tuple = ((0.001, 0.002),)
    _data_folder: str = ('./',)
    _ckpt_interval: int = (5,)
    _db_key: str = ('mongo_local',)
    _db_config: DB_CONFIG = (None,)
    _resume: bool = (True,)
    _infer_mode: bool = (False,)
    _episode_start_dt: datetime = (None,)
    plot: Plot = (None,)

    def __post_init__(self):
        self.logger = logger.getchild('main').getchild(self.__str__())
        self.logger.propagate = True
        self.dictLogger = dictLogger

        # Number of "experiences" to store     at max
        # Num of tuples to train on.
        self.touch_gpu()

    def __repr__(self):
        return f'DPG({self.truck.TruckName}, {self.driver})'

    def __str__(self):
        return 'DPG'

    @abc.abstractmethod
    def touch_gpu(self):
        pass

    @abc.abstractmethod
    def init_checkpoint(self):
        # Actor create or restore from checkpoint
        # add checkpoints manager
        pass

    @abc.abstractmethod
    def actor_predict(self, obs, t):
        """
        Evaluate the actors given a single observations.
        Batchsize is 1.
        """
        pass

    def start_episode(self, dt: datetime):
        self.logger.info(f'Episode start at {dt}', extra=dictLogger)
        # somehow mongodb does not like microseconds in rec['plot']
        dt_milliseconds = int(dt.microsecond / 1000) * 1000
        self.episode_start_dt = dt.replace(microsecond=dt_milliseconds)

        self.plot = Plot(
            character=self.truck.TruckName,
            driver=self.driver,
            when=self.episode_start_dt,
            tz=str(self.truck.tz),
            where=self.truck.Location,
            state_specs={
                'observation_specs': [
                    {'velocity_unit': 'kmph'},
                    {'thrust_unit': 'percentage'},
                    {'brake_unit': 'percentage'},
                ],
                'unit_number': self.truck.CloudUnitNumber,  # 4
                'unit_duration': self.truck.CloudUnitDuration,  # 1s
                'frequency': self.truck.CloudSignalFrequency,  # 50 hz
            },
            action_specs={
                'action_row_number': self.truck.ActionFlashRow,
                'action_column_number': self.truck.PedalScale,
            },
            reward_specs={
                'reward_unit': 'wh',
            }
        )



    @abc.abstractmethod
    def deposit(
        self, prev_ts, prev_o_t, prev_a_t, prev_table_start, cycle_reward, o_t
    ):
        """Deposit the experience into the replay buffer."""
        pass

    @abc.abstractmethod
    def end_episode(self):
        """Deposit the whole episode of experience into the replay buffer for RDPG.
        Noting to do for DDPG"""
        pass

    @abc.abstractmethod
    def train(self):
        """
        Train the actor and critic moving network.

        return:
            tuple: (actor_loss, critic_loss)
        """
        pass

    @abc.abstractmethod
    def get_losses(self):
        """
        Get the actor and critic losses without calculating the gradients.
        """
        pass

    @abc.abstractmethod
    def soft_update_target(self):
        """
        update target networks with tiny tau value, typical value 0.001.
        done after each batch, slowly update target by Polyak averaging.
        """
        pass

    @abc.abstractmethod
    def save_ckpt(self):
        """
        save checkpoints of actor and critic
        """
        pass

    @property
    def db_key(self) -> str:
        return self._db_key

    @db_key.setter
    def db_key(self, value: str):
        raise AttributeError('db_key is read-only')

    @property
    def truck(self):
        return self._truck

    @truck.setter
    def truck(self, value):
        raise AttributeError('truck is read-only')

    @property
    def driver(self):
        return self._driver

    @driver.setter
    def driver(self, value):
        raise AttributeError('driver is read-only')

    @property
    def num_states(self):
        return self._num_states

    @num_states.setter
    def num_states(self, value):
        raise AttributeError('num_states is read-only')

    @property
    def num_actions(self):
        return self._num_actions

    @num_actions.setter
    def num_actions(self, value):
        raise AttributeError('num_actions is read-only')

    @property
    def buffer_capacity(self):
        return self._buffer_capacity

    @buffer_capacity.setter
    def buffer_capacity(self, value):
        raise AttributeError('seq_len is read-only')

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        raise AttributeError('batch_size is read-only')

    @property
    def hidden_unitsAC(self):
        return self._hidden_unitsAC

    @hidden_unitsAC.setter
    def hidden_unitsAC(self, value):
        raise AttributeError('hidden_unitsAC is read-only')

    @property
    def action_bias(self):
        return self._action_bias

    @action_bias.setter
    def action_bias(self, value):
        raise AttributeError('action_bias is read-only')

    @property
    def n_layersAC(self):
        return self._n_layersAC

    @n_layersAC.setter
    def n_layersAC(self, value):
        raise AttributeError('n_layersAC is read-only')

    @property
    def padding_value(self):
        return self._padding_value

    @padding_value.setter
    def padding_value(self, value):
        raise AttributeError('padding_value is read-only')

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        raise AttributeError('gamma is read-only')

    @property
    def tauAC(self):
        return self._tauAC

    @tauAC.setter
    def tauAC(self, value):
        raise AttributeError('tauAC is read-only')

    @property
    def lrAC(self):
        return self._lrAC

    @lrAC.setter
    def lrAC(self, value):
        raise AttributeError('lrAC is read-only')

    @property
    def data_folder(self) -> str:
        return self._data_folder

    @data_folder.setter
    def data_folder(self, value: str):
        raise AttributeError('datafolder is read-only')

    @property
    def ckpt_interval(self):
        return self._ckpt_interval

    @ckpt_interval.setter
    def ckpt_interval(self, value):
        raise AttributeError('ckpt_interval is read-only')

    @property
    def resume(self):
        return self._resume

    @resume.setter
    def resume(self, value):
        raise AttributeError('resume is read-only')

    @property
    def infer_mode(self):
        return self._infer_mode

    @infer_mode.setter
    def infer_mode(self, value):
        raise AttributeError('infer_mode is read-only')

    @property
    def db_config(self) -> DB_CONFIG:
        return self._db_config

    @db_config.setter
    def db_config(self, value: DB_CONFIG):
        self._db_config = value

    @property
    def episode_start_dt(self) -> datetime:
        return self._episode_start_dt

    @episode_start_dt.setter
    def episode_start_dt(self, value: datetime):
        self._episode_start_dt = value
