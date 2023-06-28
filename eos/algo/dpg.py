import abc
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import pandas as pd
import re

from eos.data_io.config import Truck, trucks_by_id, get_db_config, Driver, drivers_by_id
from eos.data_io.struct import (
    StateUnitCodes,
    ObservationMeta,
    StateSpecs,
    ActionSpecs,
    get_filemeta_config,
)
from eos.data_io.buffer import Buffer, MongoBuffer, ArrowBuffer


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
    # or 'veos:asdf@localhost:27017' for database access
    # or 'recipe.ini': when combined with _data_folder, indicate the configparse ini file for local file access
    _observation_meta: Optional[ObservationMeta] = None
    _episode_start_dt: datetime = None
    _truck: Truck = trucks_by_id['VB7']
    _driver: Driver = drivers_by_id['zheng-longfei']
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
    _observations: list[pd.DataFrame] = field(default_factory=list[pd.DataFrame])
    _torque_table_row_names: list[str] = field(default_factory=list[str])
    _epi_no: int = 0

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

        #  init observation meta info object,
        #  episode start time will be updated for each episode, for now it is the time when the algo is created

        self.observation_meta = ObservationMeta(
            state_specs=StateSpecs(
                state_unit_codes=StateUnitCodes(
                    velocity_unit_code='kph',
                    thrust_unit_code='pct',
                    brake_unit_code='pct',
                ),
                unit_number=self.truck.cloud_unit_number,  # 4
                unit_duration=self.truck.cloud_unit_duration,  # 1s
                frequency=self.truck.cloud_signal_frequency,  # 50 hz
            ),
            action_specs=ActionSpecs(
                action_unit_code='nm',
                action_row_number=self.truck.action_flashrow,
                action_column_number=len(self.truck.pedal_scale),
            ),
            reward_specs={
                'reward_unit': 'wh',
            },
            site=self.truck.site,
        )

        (
            self.num_states,
            self.num_actions,
        ) = self.observation_meta.get_number_of_states_actions()

        self.torque_table_row_names = (
            self.observation_meta.get_torque_table_row_names()
        )  # part of the action MultiIndex
        login_pattern = re.compile(
            r'^[A-Za-z]\w*:\w+@\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d{1,5}'
        )
        recipe_pattern = re.compile(r'^[A-Za-z]\w*\.ini$')
        # if pool_key is an url or a mongodb name
        if 'mongo' in self.pool_key.lower() or login_pattern.match(self.pool_key):
            db_config = get_db_config(self.pool_key)
            self.buffer = MongoBuffer(  # choose item type: Record/Episode
                db_config=db_config,
                batch_size=self.batch_size,
                driver=self.driver,
                truck=self.truck,
                meta=self.observation_meta,
                torque_table_row_names=self.torque_table_row_names,
            )
        elif self.pool_key is None or recipe_pattern.match(
            self.pool_key
        ):  # if pool_key is an ini filename, use parquet as pool
            recipe = get_filemeta_config(
                data_folder=self.data_folder,
                config_file=self.pool_key,
                meta=self.observation_meta,
            )
            self.buffer = ArrowBuffer(
                recipe=recipe,
                batch_size=self.batch_size,
                driver=self.driver,
                truck=self.truck,
                meta=self.observation_meta,
                torque_table_row_names=self.torque_table_row_names,
            )
        else:
            raise ValueError(
                f'pool_key {self.pool_key} is not a valid mongodb login string nor an ini filename.'
            )

    def __repr__(self):
        return f'DPG({self.truck.vid}, {self.driver.pid})'

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
    def actor_predict(self, state: pd.DataFrame, t: int):
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

        self.observations = []

    @abc.abstractmethod
    def deposit(
        self,
        timestamp: pd.Timestamp,
        state: pd.DataFrame,
        action: pd.DataFrame,
        reward: pd.DataFrame,
        nstate: pd.DataFrame,
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
    def observation_meta(self) -> ObservationMeta:
        return self._observation_meta

    @observation_meta.setter
    def observation_meta(self, value: ObservationMeta):
        self._observation_meta = value

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

    @property
    def observations(self) -> list[pd.DataFrame]:
        return self._observations

    @observations.setter
    def observations(self, value: list[pd.DataFrame]):
        self._observations = value

    @property
    def epi_no(self) -> int:
        return self._epi_no

    @epi_no.setter
    def epi_no(self, value: int):
        self._epi_no = value

    @property
    def torque_table_row_names(self) -> list[str]:
        return self._torque_table_row_names

    @torque_table_row_names.setter
    def torque_table_row_names(self, value: list[str]):
        self._torque_table_row_names = value
