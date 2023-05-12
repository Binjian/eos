import abc
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import re

import tensorflow as tf

from eos.data_io.config import Truck, trucks_by_name, get_db_config
from eos.data_io.struct import (
    ObservationSpecs,
    Plot,
    RecordArr,
    RecordDoc,
    EpisodeDoc,
    EpisodeArr,
    get_filepool_config,
)
from eos.data_io.buffer import Buffer, DBBuffer, FileBuffer


"""Base class for differentiable policy gradient methods."""


@dataclass
class DPG(abc.ABC):
    """Base class for differentiable policy gradient methods."""

    _coll_type: str = 'RECORD'
    _buffer: Optional[
        Buffer
    ] = None  # as last of non-default parameters, so that derived class can override with default
    _data_folder: str = './'
    _pool_key: str = 'mongo_local'  # 'mongo_***'
    # or 'veos:asdf@localhost:27017' for databse access
    # or 'recipe.ini': when combined with _data_folder, indicate the configparse ini file for local file access
    _plot: Optional[Plot] = None
    _episode_start_dt: datetime = None
    _truck: Truck = trucks_by_name['VB7']
    _driver: str = 'longfei'
    _num_states: int = 600
    _num_actions: int = 68
    _buffer_capacity: int = 10000
    _batch_size: int = 4
    _hidden_units_ac: tuple = (256, 16, 32)
    _action_bias: float = 0.0
    _n_layers_ac: tuple = (2, 2)
    _padding_value: float = 0
    _gamma: float = 0.99
    _tau_ac: tuple = (0.005, 0.005)
    _lr_ac: tuple = (0.001, 0.002)
    _ckpt_interval: int = 5
    _resume: bool = True
    _infer_mode: bool = False

    def __post_init__(self):
        """
        Initialize the DPG object.
        Heavy lifting of data interface (buffer, pool) for both DDPG and RDPG
        """
        # pass
        # Number of "experiences" to store     at max
        # Num of tuples to train on.

        dt = datetime.now()
        dt_milliseconds = int(dt.microsecond / 1000) * 1000
        self.episode_start_dt = dt.replace(microsecond=dt_milliseconds)

        #  init plot object, episode start time will be updated for each episode, for now it is the time when the algo is created
        self.plot = (
            Plot(  # self.plot is a Plot object generated from realtime truck object
                character=self.truck.TruckName,
                driver=self.driver,
                when=self.episode_start_dt,
                tz=str(self.truck.tz),
                where=self.truck.Location,
                state_specs={
                    'observation_specs': ObservationSpecs(
                        velocity_unit='kph',
                        thrust_unit='pct',
                        brake_unit='pct',
                    ),
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
                },
            )
        )
        self.num_states, self.num_actions = self.plot.get_number_of_states_actions()

        login_pattern = re.compile(
            r'^[A-Za-z]\w*:\w+@\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d{1,5}'
        )
        recipe_pattern = re.compile(r'^[A-Za-z]\w*\.ini$')
        # if pool_key is an url or a mongodb name
        if 'mongo' in self.pool_key.lower() or login_pattern.match(self.pool_key):
            db_config = get_db_config(self.pool_key)
            db_config._replace(type='RECORD')  # update the db_config type to record
            if 'RECORD' in self.coll_type.upper():
                self.buffer = DBBuffer[RecordDoc](  # choose item type: Record/Episode
                    plot=self.plot,
                    db_config=db_config,
                    batch_size=self.batch_size,
                )
            elif 'EPISODE' in self.coll_type.upper():
                self.buffer = DBBuffer[EpisodeDoc](  # choose item type: Record/Episode
                    plot=self.plot,
                    db_config=db_config,
                    batch_size=self.batch_size,
                )
            else:
                raise ValueError(
                    f'coll_type {self.coll_type} is not a valid collection type. It should be RECORD or EPISODE.'
                )
        elif self.pool_key is None or recipe_pattern.match(
            self.pool_key
        ):  # if pool_key is an ini filename, use local files as pool
            if 'RECORD' in self.coll_type.upper():
                recipe = get_filepool_config(
                    data_folder=self.data_folder,
                    config_file=self.pool_key,
                    coll_type='RECORD',
                    plot=self.plot,
                )
                self.buffer = FileBuffer[RecordDoc](
                    plot=self.plot,
                    recipe=recipe,
                    batch_size=self.batch_size,
                )
            elif 'EPISODE' in self.coll_type.upper():
                recipe = get_filepool_config(
                    data_folder=self.data_folder,
                    config_file=self.pool_key,
                    coll_type='EPISODE',
                    plot=self.plot,
                )
                self.buffer = FileBuffer[EpisodeDoc](
                    plot=self.plot,
                    recipe=recipe,
                    batch_size=self.batch_size,
                    coll_type='EPISODE',  # choose item type: Record/Episode
                )
        else:
            raise ValueError(
                f'pool_key {self.pool_key} is not a valid mongodb login string nor an ini filename.'
            )

    def __repr__(self):
        return f'DPG({self.truck.TruckName}, {self.driver})'

    def __str__(self):
        return 'DPG'

    def __hash__(self):
        return hash(self.__repr__())

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
        batch_size is 1.
        """
        pass

    def start_episode(self, dt: datetime):
        # self.logger.info(f'Episode start at {dt}', extra=dictLogger)
        # somehow mongodb does not like microseconds in rec['plot']
        dt_milliseconds = int(dt.microsecond / 1000) * 1000
        self.episode_start_dt = dt.replace(microsecond=dt_milliseconds)

        self.plot.when = self.episode_start_dt  # only update when an episode starts

    @abc.abstractmethod
    def deposit(
        self,
        prev_ts: tf.Tensor,
        prev_o_t: tf.Tensor,
        prev_a_t: tf.Tensor,
        prev_table_start: int,
        cycle_reward: float,
        o_t: tf.Tensor,
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
    def pool_key(self) -> str:
        return self._pool_key

    @pool_key.setter
    def pool_key(self, value: str):
        raise AttributeError('pool_key is read-only')

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
    def hidden_units_ac(self):
        return self._hidden_units_ac

    @hidden_units_ac.setter
    def hidden_units_ac(self, value):
        raise AttributeError('hidden_units_ac is read-only')

    @property
    def action_bias(self):
        return self._action_bias

    @action_bias.setter
    def action_bias(self, value):
        raise AttributeError('action_bias is read-only')

    @property
    def n_layers_ac(self):
        return self._n_layers_ac

    @n_layers_ac.setter
    def n_layers_ac(self, value):
        raise AttributeError('n_layers_ac is read-only')

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
    def tau_ac(self):
        return self._tau_ac

    @tau_ac.setter
    def tau_ac(self, value):
        raise AttributeError('tau_ac is read-only')

    @property
    def lr_ac(self):
        return self._lr_ac

    @lr_ac.setter
    def lr_ac(self, value):
        raise AttributeError('lr_ac is read-only')

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
    def episode_start_dt(self) -> datetime:
        return self._episode_start_dt

    @episode_start_dt.setter
    def episode_start_dt(self, value: datetime):
        self._episode_start_dt = value

    @property
    def plot(self) -> Plot:
        return self._plot

    @plot.setter
    def plot(self, value: Plot):
        self._plot = value

    @property
    def buffer(self) -> Buffer:
        return self._buffer

    @buffer.setter
    def buffer(self, value: Buffer):
        self._buffer = value

    @property
    def coll_type(self) -> str:
        return self._coll_type

    @coll_type.setter
    def coll_type(self, value: str):
        self._coll_type = value
