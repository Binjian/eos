import abc
import pandas as pd
from datetime import datetime
from eos.data_io.buffer import DaskBuffer as DaskBuffer, MongoBuffer as MongoBuffer
from eos.data_io.config import (
    Driver as Driver,
    RE_DBKEY as RE_DBKEY,
    Truck as Truck,
    TruckInCloud as TruckInCloud,
    get_db_config as get_db_config,
)
from eos.data_io.struct import (
    ObservationMetaCloud as ObservationMetaCloud,
    ObservationMetaField as ObservationMetaField,
    RE_RECIPEKEY as RE_RECIPEKEY,
    get_filemeta_config as get_filemeta_config,
)
from typing import Union
from utils import HyperParam as HyperParam

class DPG(abc.ABC, metaclass=abc.ABCMeta):
    def __post_init__(self) -> None: ...
    def __hash__(self): ...
    @abc.abstractmethod
    def touch_gpu(self): ...
    @abc.abstractmethod
    def init_checkpoint(self): ...
    @abc.abstractmethod
    def actor_predict(self, state: pd.Series, t: int): ...
    def start_episode(self, dt: datetime): ...
    def deposit(
        self,
        timestamp: pd.Timestamp,
        state: pd.Series,
        action: pd.Series,
        reward: pd.Series,
        nstate: pd.Series,
    ): ...
    def end_episode(self) -> None: ...
    def deposit_episode(self) -> None: ...
    @abc.abstractmethod
    def train(self): ...
    @abc.abstractmethod
    def get_losses(self): ...
    @abc.abstractmethod
    def soft_update_target(self): ...
    @abc.abstractmethod
    def save_ckpt(self): ...
    @property
    def pool_key(self) -> str: ...
    @property
    def truck(self): ...
    @property
    def driver(self): ...
    @property
    def data_folder(self) -> str: ...
    @property
    def resume(self): ...
    @property
    def infer_mode(self): ...
    @property
    def episode_start_dt(self) -> datetime: ...
    @property
    def observation_meta(self) -> Union[ObservationMetaCloud, ObservationMetaField]: ...
    @property
    def buffer(self) -> Union[MongoBuffer, DaskBuffer]: ...
    @property
    def coll_type(self) -> str: ...
    @property
    def observations(self) -> list[pd.Series]: ...
    @property
    def epi_no(self) -> int: ...
    @property
    def torque_table_row_names(self) -> list[str]: ...
    @property
    def hyper_param(self) -> HyperParam: ...
    def __init__(
        self,
        *,
        _truck,
        _driver,
        _buffer,
        _coll_type,
        _hyper_param,
        _pool_key,
        _data_folder,
        _infer_mode,
        _observation_meta,
        _episode_start_dt,
        _resume,
        _observations,
        _torque_table_row_names,
        _epi_no
    ) -> None: ...
