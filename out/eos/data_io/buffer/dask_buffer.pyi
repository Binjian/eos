from configparser import ConfigParser
from typing import Tuple

import numpy as np
import pandas as pd
from _typeshed import Incomplete

from eos.data_io.buffer import Buffer as Buffer
from eos.data_io.config import Driver as Driver
from eos.data_io.config import Truck as Truck
from eos.data_io.config import drivers_by_id as drivers_by_id
from eos.data_io.config import trucks_by_id as trucks_by_id
from eos.data_io.pool import AvroPool as AvroPool
from eos.data_io.pool import DaskPool as DaskPool
from eos.data_io.pool import ParquetPool as ParquetPool
from eos.data_io.struct import ObservationMeta as ObservationMeta
from eos.data_io.struct import PoolQuery as PoolQuery
from eos.data_io.struct import veos_lifetime_end_date as veos_lifetime_end_date
from eos.data_io.struct import \
    veos_lifetime_start_date as veos_lifetime_start_date
from eos.utils import dictLogger as dictLogger
from eos.utils import logger as logger
from eos.utils.eos_pandas import \
    decode_episode_dataframes_to_padded_arrays as \
    decode_episode_dataframes_to_padded_arrays

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
