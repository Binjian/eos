from configparser import ConfigParser
from datetime import datetime
from typing import Dict, List, Optional, Tuple, TypeVar, Union

import pandas as pd
from _typeshed import Incomplete
from bson import ObjectId as ObjectId
from pydantic import BaseModel
from typing_extensions import TypedDict

from eos.data_io.config import CloudMixin as CloudMixin
from eos.data_io.config import KvaserMixin as KvaserMixin

veos_lifetime_start_date: datetime
veos_lifetime_end_date: datetime

class StateUnitCodes(BaseModel):
    velocity_unit_code: str
    thrust_unit_code: str
    brake_unit_code: str

class StateSpecs(BaseModel):
    interface: str
    state_unit_codes: StateUnitCodes
    state_number: int
    unit_number_per_state: float
    unit_duration: float
    frequency: float

class StateSpecsCloud(CloudMixin, StateSpecs):
    interface: str
    state_number: int
    unit_number_per_state: Incomplete
    unit_duration: Incomplete
    def __post_init__(self) -> None: ...

class StateSpecsKvaser(KvaserMixin, StateSpecs):
    interface: str
    state_number: int
    unit_number_per_state: Incomplete
    unit_duration: Incomplete
    def __post_init__(self) -> None: ...

class ActionSpecs(BaseModel):
    action_unit_code: str
    action_row_number: int
    action_column_number: int

class RewardSpecs(BaseModel):
    reward_unit_code: str
    reward_number: int

class ObservationMeta(BaseModel):
    state_specs: StateSpecs
    action_specs: ActionSpecs
    reward_specs: RewardSpecs
    site: str
    def get_number_of_states(self) -> float: ...
    def get_number_of_actions(self) -> int: ...
    def get_number_of_states_actions(self) -> Tuple[float, int]: ...
    def have_same_meta(self, meta_to_compare: ObservationMeta): ...
    def get_torque_table_row_names(self) -> List[str]: ...

class ObservationMetaCloud(ObservationMeta, BaseModel):
    state_specs: StateSpecsCloud

class ObservationMetaField(ObservationMeta, BaseModel):
    state_specs: StateSpecsKvaser

class DataFrameDoc(TypedDict):
    timestamp: datetime
    meta: dict
    observation: dict
ItemT = TypeVar('ItemT', Dict, pd.DataFrame)

class PoolQuery(BaseModel):
    vehicle: str
    driver: str
    episodestart_start: datetime
    episodestart_end: datetime
    timestamp_start: Optional[datetime]
    timestamp_end: Optional[datetime]

RE_RECIPEKEY: Incomplete

def get_filemeta_config(data_folder: str, config_file: Optional[str], meta: Union[ObservationMetaCloud, ObservationMetaField], coll_type: str) -> ConfigParser: ...
