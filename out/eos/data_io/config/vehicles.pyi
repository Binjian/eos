from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
from _typeshed import Incomplete
from ordered_set import OrderedSet

from eos.data_io.struct import timezones as timezones

PEDAL_SCALES: Incomplete
SPEED_SCALES_MULE: Incomplete
SPEED_SCALES_VB: Incomplete
TRIANGLE_TEST_CASE_TARGET_VELOCITIES: Incomplete
TruckCat: Incomplete
Maturity: Incomplete
RE_VIN: str

class OperationHistory:
    site: str
    date_range: pd.DatetimeIndex
    def __init__(self, site, date_range) -> None: ...

class KvaserMixin:
    kvaser_observation_number: int
    kvaser_observation_frequency: float
    kvaser_countdown: int
    def __init__(
        self, kvaser_observation_number, kvaser_observation_frequency, kvaser_countdown
    ) -> None: ...

class CloudMixin:
    cloud_signal_frequency: float
    cloud_gear_frequency: float
    cloud_unit_duration: float
    cloud_unit_number: int
    def __init__(
        self,
        cloud_signal_frequency,
        cloud_gear_frequency,
        cloud_unit_duration,
        cloud_unit_number,
    ) -> None: ...

class Truck:
    vid: str
    vin: str
    plate: str
    maturity: str
    site: str
    operation_history: list[OperationHistory]
    interface: str
    tz: Optional[ZoneInfo]
    tbox_id: Optional[str]
    pedal_scale: tuple
    speed_scale: tuple
    observation_number: int
    torque_budget: int
    torque_lower_bound: float
    torque_upper_bound: float
    torque_bias: float
    torque_table_row_num_flash: int
    cat: OrderedSet
    def __post_init__(self) -> None: ...
    @property
    def torque_flash_numel(self): ...
    @property
    def torque_full_numel(self): ...
    @property
    def observation_numel(self): ...
    @property
    def observation_length(self): ...
    @property
    def observation_sampling_rate(self): ...
    @property
    def observation_duration(self): ...
    @property
    def torque_table_row_num(self): ...
    @property
    def torque_table_col_num(self): ...
    def __init__(
        self,
        vid,
        vin,
        plate,
        maturity,
        site,
        operation_history,
        interface,
        tz,
        tbox_id,
        pedal_scale,
        _torque_table_col_num,
        speed_scale,
        _torque_table_row_num,
        observation_number,
        torque_budget,
        torque_lower_bound,
        torque_upper_bound,
        torque_bias,
        torque_table_row_num_flash,
        cat,
        _torque_flash_numel,
        _torque_full_numel,
        _observation_numel,
        _observation_length,
        _observation_sampling_rate,
        _observation_duration,
    ) -> None: ...

class TruckInCloud(CloudMixin, Truck):
    interface: str
    observation_length: Incomplete
    observation_numel: Incomplete
    observation_sampling_rate: Incomplete
    observation_duration: Incomplete
    torque_table_row_num_flash: int
    def __post_init__(self) -> None: ...
    def __init__(
        self,
        vid,
        vin,
        plate,
        maturity,
        site,
        operation_history,
        interface,
        tz,
        tbox_id,
        pedal_scale,
        _torque_table_col_num,
        speed_scale,
        _torque_table_row_num,
        observation_number,
        torque_budget,
        torque_lower_bound,
        torque_upper_bound,
        torque_bias,
        torque_table_row_num_flash,
        cat,
        _torque_flash_numel,
        _torque_full_numel,
        _observation_numel,
        _observation_length,
        _observation_sampling_rate,
        _observation_duration,
        cloud_signal_frequency,
        cloud_gear_frequency,
        cloud_unit_duration,
        cloud_unit_number,
    ) -> None: ...

class TruckInField(KvaserMixin, Truck):
    interface: str
    observation_length: Incomplete
    observation_numel: Incomplete
    observation_sampling_rate: Incomplete
    observation_duration: Incomplete
    torque_table_row_num_flash: int
    def __post_init__(self) -> None: ...
    def __init__(
        self,
        vid,
        vin,
        plate,
        maturity,
        site,
        operation_history,
        interface,
        tz,
        tbox_id,
        pedal_scale,
        _torque_table_col_num,
        speed_scale,
        _torque_table_row_num,
        observation_number,
        torque_budget,
        torque_lower_bound,
        torque_upper_bound,
        torque_bias,
        torque_table_row_num_flash,
        cat,
        _torque_flash_numel,
        _torque_full_numel,
        _observation_numel,
        _observation_length,
        _observation_sampling_rate,
        _observation_duration,
        kvaser_observation_number,
        kvaser_observation_frequency,
        kvaser_countdown,
    ) -> None: ...

trucks: Incomplete
trucks_all: Incomplete
trucks_by_id: Incomplete
trucks_by_vin: Incomplete
