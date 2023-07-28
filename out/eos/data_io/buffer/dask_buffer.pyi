import numpy as np
import pandas as pd
from _typeshed import Incomplete
from configparser import ConfigParser
from eos.data_io.buffer import Buffer as Buffer
from eos.data_io.config import Driver as Driver, Truck as Truck, drivers_by_id as drivers_by_id, trucks_by_id as trucks_by_id
from eos.data_io.pool import AvroPool as AvroPool, DaskPool as DaskPool, ParquetPool as ParquetPool
from eos.data_io.struct import ObservationMeta as ObservationMeta, PoolQuery as PoolQuery, veos_lifetime_end_date as veos_lifetime_end_date, veos_lifetime_start_date as veos_lifetime_start_date
from eos.utils import dictLogger as dictLogger, logger as logger
from eos.utils.eos_pandas import decode_episode_dataframes_to_padded_arrays as decode_episode_dataframes_to_padded_arrays
from typing import Tuple

class DaskBuffer(Buffer[pd.DataFrame]):
    pool: DaskPool
    recipe: ConfigParser
    batch_size: int
    driver: Driver
    truck: Truck
    buffer_count: int
    query: PoolQuery
    meta: ObservationMeta
    torque_table_row_names: list[str]
    logger: Incomplete
    def __post_init__(self) -> None: ...
    def load(self) -> None: ...
    def decode_batch_records(self, batch: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
    def sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
    def close(self) -> None: ...
    def __init__(self, *, pool, batch_size, buffer_count, recipe, driver, truck, query, meta, torque_table_row_names) -> None: ...
