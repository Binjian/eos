from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime
from typing import Union, ClassVar
import pandas as pd
import re

from eos.data_io.config import (
    Truck,
    TruckInCloud,
    trucks_by_id,
    get_db_config,
    RE_DBKEY,
    Driver,
)
from eos.data_io.struct import (
    ObservationMetaCloud,
    ObservationMetaField,
    get_filemeta_config,
    RE_RECIPEKEY,
)
from utils import HyperParamRDPG, HyperParamDDPG  # type: ignore
from eos.data_io.buffer import MongoBuffer, DaskBuffer
from eos.utils.eos_pandas import encode_episode_dataframe_from_series


"""Base class for differentiable policy gradient methods."""


@dataclass(kw_only=True)
class DPG(abc.ABC):
    """Base class for differentiable policy gradient methods."""

    _truck_type: ClassVar[Truck] = trucks_by_id[
        "default"
    ]  # class attribute for default truck properties

    _truck: Truck
    _driver: Driver
    _buffer: Union[MongoBuffer, DaskBuffer] = field(default_factory=MongoBuffer)
    # as last of non-default parameters, so that derived class can override with default
    _coll_type: str = (
        "RECORD"  # or 'EPISODE', used for create different buffer and pool
    )
    _hyper_param: Union[HyperParamDDPG, HyperParamRDPG] = HyperParamDDPG('DDPG')
    _pool_key: str = "mongo_local"  # 'mongo_***'
    # or 'veos:asdf@localhost:27017' for database access
    # or 'recipe.ini': when combined with _data_folder, indicate the configparse ini file for local file access
    _data_folder: str = "./"
    _infer_mode: bool = False
    # Following are derived from above
    _observation_meta: Union[
        ObservationMetaCloud, ObservationMetaField
    ] = ObservationMetaCloud()
    _episode_start_dt: datetime = datetime.now()
    _resume: bool = True
    _observations: list[pd.Series] = field(default_factory=list[pd.Series])
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

        if isinstance(self.truck, TruckInCloud):
            self.observation_meta = ObservationMetaCloud()  # use default
        else:  # Kvaser
            self.observation_meta = ObservationMetaField()  # use default

        (
            self.truck.observation_numel,
            self.truck.torque_flash_numel,
        ) = self.observation_meta.get_number_of_states_actions()

        self.torque_table_row_names = (
            self.observation_meta.get_torque_table_row_names()
        )  # part of the action MultiIndex
        login_pattern = re.compile(RE_DBKEY)
        recipe_pattern = re.compile(RE_RECIPEKEY)
        # if pool_key is an url or a mongodb name
        if "mongo" in self.pool_key.lower() or login_pattern.match(self.pool_key):
            # TODO coll_type needs to be passed in for differentiation between RECORD and EPISODE
            db_config = get_db_config(self.pool_key, self.coll_type)
            self.buffer = MongoBuffer(  # choose item type: Record/Episode
                db_config=db_config,
                batch_size=self.hyper_param.BatchSize,
                driver=self.driver,
                truck=self.truck,
                meta=self.observation_meta,
                buffer_count=0,  # will be updated during initialization of Buffer
                torque_table_row_names=self.torque_table_row_names,
            )
        elif self.pool_key is None or recipe_pattern.match(
            self.pool_key
        ):  # if pool_key is an ini filename, use parquet as pool
            recipe = get_filemeta_config(
                data_folder=self.data_folder,
                config_file=self.pool_key,
                meta=self.observation_meta,
                coll_type=self.coll_type,
            )
            self.buffer = DaskBuffer(
                recipe=recipe,
                batch_size=self.hyper_param.BatchSize,
                driver=self.driver,
                truck=self.truck,
                meta=self.observation_meta,
                buffer_count=0,
                torque_table_row_names=self.torque_table_row_names,
            )
        else:
            raise ValueError(
                f"pool_key {self.pool_key} is not a valid mongodb login string nor an ini filename."
            )

    def __repr__(self):
        return f"DPG({self.truck.vid}, {self.driver.pid})"

    def __str__(self):
        return "DPG"

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
    def actor_predict(self, state: pd.Series):
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

        self.observations: list[
            pd.Series
        ] = []  # create a new empty list for each episode

    # @abc.abstractmethod
    def deposit(
        self,
        timestamp: pd.Timestamp,
        state: pd.Series,
        action: pd.Series,
        reward: pd.Series,
        nstate: pd.Series,
    ):
        """
        Deposit the experience into the replay buffer.
        state: pd.Series [brake row -> thrust row  -> timestep row -> velocity row ]
        action: pd.Series [r0, r1, r2, ... rows -> speed row -> throttle row-> (flash) timestep row ]
        reward: pd.Series [timestep row -> work row]
        nstate: like state
        """

        # Create MultiIndex
        timestamp_ser = pd.Series([timestamp], name="timestamp")
        timestamp_ser.index = pd.MultiIndex.from_product(
            [timestamp_ser.index, [0]], names=["rows", "idx"]
        )
        timestamp_index = (timestamp_ser.name, "", 0)  # triple index (name, row, idx)
        state_index = [(state.name, *i) for i in state.index]
        reward_index = [(reward.name, *i) for i in reward.index]
        action_index = [(action.name, *i) for i in action.index]
        nstate_index = [(nstate.name, *i) for i in nstate.index]

        multiindex = pd.MultiIndex.from_tuples(
            [timestamp_index, *state_index, *action_index, *reward_index, *nstate_index]
        )
        observation_list = [timestamp_ser, state, action, reward, nstate]
        observation = pd.concat(
            observation_list
        )  # concat Series along MultiIndex, still a Series
        observation.index = multiindex  # each observation is a series for the quadruple (s,a,r,s') with a MultiIndex
        self.observations.append(
            observation
        )  # each observation is a series for the quadruple (s,a,r,s')

    # @abc.abstractmethod
    def end_episode(self):
        """
        Deposit the whole episode of experience into the replay buffer for DPG.
        """
        self.deposit_episode()
        self.epi_no += 1

    def deposit_episode(self):
        """
        Deposit the whole episode of experience into the replay buffer for DPG.
        """

        episode = encode_episode_dataframe_from_series(
            self.observations,
            self.torque_table_row_names,
            self.episode_start_dt,
            self.truck.vid,
            self.driver.pid,
        )
        self.buffer.store(episode)

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
        raise AttributeError("pool_key is read-only")

    @property
    def truck(self):
        return self._truck

    @truck.setter
    def truck(self, value):
        raise AttributeError("truck is read-only")

    @property
    def driver(self):
        return self._driver

    @driver.setter
    def driver(self, value):
        raise AttributeError("driver is read-only")

    @property
    def data_folder(self) -> str:
        return self._data_folder

    @data_folder.setter
    def data_folder(self, value: str):
        raise AttributeError("datafolder is read-only")

    @property
    def resume(self):
        return self._resume

    @resume.setter
    def resume(self, value):
        raise AttributeError("resume is read-only")

    @property
    def infer_mode(self):
        return self._infer_mode

    @infer_mode.setter
    def infer_mode(self, value):
        raise AttributeError("infer_mode is read-only")

    @property
    def episode_start_dt(self) -> datetime:
        return self._episode_start_dt

    @episode_start_dt.setter
    def episode_start_dt(self, value: datetime):
        self._episode_start_dt = value

    @property
    def observation_meta(self) -> Union[ObservationMetaCloud, ObservationMetaField]:
        return self._observation_meta

    @observation_meta.setter
    def observation_meta(
        self, value: Union[ObservationMetaCloud, ObservationMetaField]
    ):
        self._observation_meta = value

    @property
    def buffer(self) -> Union[MongoBuffer, DaskBuffer]:
        return self._buffer

    @buffer.setter
    def buffer(self, value: Union[MongoBuffer, DaskBuffer]):
        self._buffer = value

    @property
    def coll_type(self) -> str:
        return self._coll_type

    @coll_type.setter
    def coll_type(self, value: str):
        self._coll_type = value

    @property
    def observations(self) -> list[pd.Series]:
        return self._observations

    @observations.setter
    def observations(self, value: list[pd.Series]):
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

    @property
    def hyper_param(self) -> Union[HyperParamDDPG, HyperParamRDPG]:
        return self._hyper_param

    @hyper_param.setter
    def hyper_param(self, value: Union[HyperParamDDPG, HyperParamRDPG]):
        self._hyper_param = value
