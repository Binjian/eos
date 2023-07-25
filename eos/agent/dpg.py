from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Union
import pandas as pd
import re

from eos.data_io.config import (
    Truck,
    get_db_config,
    RE_DBKEY,
    Driver,
)
from eos.data_io.struct import (
    StateUnitCodes,
    ObservationMeta,
    StateSpecs,
    ActionSpecs,
    get_filemeta_config,
    RE_RECIPEKEY,
)
from utils import hyper_param_by_name, HYPER_PARAM  # type: ignore
from eos.data_io.buffer import Buffer, MongoBuffer, DaskBuffer


"""Base class for differentiable policy gradient methods."""


@dataclass(kw_only=True)
class DPG(abc.ABC):
    """Base class for differentiable policy gradient methods."""

    _truck: Truck
    _driver: Driver
    _coll_type: str = (
        "RECORD"  # or 'EPISODE', used for create different buffer and pool
    )
    _hyper_param: HYPER_PARAM = hyper_param_by_name["DEFAULT"]
    _pool_key: str = "mongo_local"  # 'mongo_***'
    # or 'veos:asdf@localhost:27017' for database access
    # or 'recipe.ini': when combined with _data_folder, indicate the configparse ini file for local file access
    _data_folder: str = "./"
    _infer_mode: bool = False
    # Following are derived from above
    _buffer: Optional[
        Buffer
    ] = None  # as last of non-default parameters, so that derived class can override with default
    _observation_meta: Optional[ObservationMeta] = None
    _episode_start_dt: Optional[datetime] = None
    _resume: bool = True
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
                    velocity_unit_code="kph",
                    thrust_unit_code="pct",
                    brake_unit_code="pct",
                ),
                state_number=3,  # velocity, thrust, and brake
                unit_number=self.truck.cloud_unit_number,  # 4
                unit_duration=self.truck.cloud_unit_duration,  # 1s
                frequency=self.truck.cloud_signal_frequency,  # 50 hz
            ),
            action_specs=ActionSpecs(
                action_unit_code="nm",
                action_row_number=self.truck.torque_table_row_num_flash,
                action_column_number=len(self.truck.pedal_scale),
            ),
            reward_specs={
                "reward_unit": "wh",
            },
            site=self.truck.site,
        )

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
            # TODO coll_type needs to be passed in for differentiantion between RECORD and EPISODE
            db_config = get_db_config(self.pool_key, self.coll_type)
            self.buffer = MongoBuffer(  # choose item type: Record/Episode
                db_config=db_config,
                batch_size=self.hyper_param.BatchSize,
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
                coll_type=self.coll_type,
            )
            self.buffer = DaskBuffer(
                recipe=recipe,
                batch_size=self.hyper_param.BatchSize,
                driver=self.driver,
                truck=self.truck,
                meta=self.observation_meta,
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
    def actor_predict(self, state: pd.Series, t: int):
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
        ts = pd.Series([timestamp], name="timestamp")
        ts.index = pd.MultiIndex.from_product([ts.index, [0]], names=["rows", "idx"])
        timestamp_index = (ts.name, "", 0)  # triple index (name, row, idx)
        state_index = [(state.name, *i) for i in state.index]
        reward_index = [(reward.name, *i) for i in reward.index]
        action_index = [(action.name, *i) for i in action.index]
        nstate_index = [(nstate.name, *i) for i in nstate.index]

        multiindex = pd.MultiIndex.from_tuples(
            [timestamp_index, *state_index, *action_index, *reward_index, *nstate_index]
        )
        observation_list = [timestamp, state, action, reward, nstate]
        observation = pd.concat(observation_list)  # concat Series along MultiIndex,
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

        episode = pd.concat(
            self.observations, axis=1
        ).transpose()  # concat along columns and transpose to DataFrame, columns not sorted as (s,a,r,s')
        episode.columns.name = ["tuple", "rows", "idx"]
        episode.set_index(("timestamp", "", 0), append=False, inplace=True)
        episode.index.name = "timestamp"
        # episode.sort_index(inplace=True)

        # convert columns types to float where necessary
        state_cols_float = [("state", col) for col in ["brake", "thrust", "velocity"]]
        action_cols_float = [
            ("action", col)
            for col in [*self.torque_table_row_names, "speed", "throttle"]
        ]
        reward_cols_float = [("reward", "work")]
        nstate_cols_float = [("nstate", col) for col in ["brake", "thrust", "velocity"]]
        for col in (
            action_cols_float + state_cols_float + reward_cols_float + nstate_cols_float
        ):
            episode[col[0], col[1]] = episode[col[0], col[1]].astype(
                "float"
            )  # float16 not allowed in parquet

        # Create MultiIndex for the episode, in the order 'episodestart', 'vehicle', 'driver'
        episode = pd.concat(
            [episode],
            keys=[pd.to_datetime(self.episode_start_dt)],
            names=["episodestart"],
        )
        episode = pd.concat([episode], keys=[self.driver.pid], names=["driver"])
        episode = pd.concat([episode], keys=[self.truck.vid], names=["vehicle"])
        episode.sort_index(inplace=True)  # sorting in the time order of timestamps

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

    @property
    def hyper_param(self) -> HYPER_PARAM:
        return self._hyper_param

    @hyper_param.setter
    def hyper_param(self, value: HYPER_PARAM):
        self._hyper_param = value
