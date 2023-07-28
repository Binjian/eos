import numpy as np
import pandas as pd
from _typeshed import Incomplete
from eos.data_io.buffer import Buffer as Buffer
from eos.data_io.config import DBConfig as DBConfig, Driver as Driver, Truck as Truck, db_config_servers_by_name as db_config_servers_by_name, drivers_by_id as drivers_by_id, trucks_by_id as trucks_by_id
from eos.data_io.pool import MongoPool as MongoPool
from eos.data_io.struct import ObservationMeta as ObservationMeta, PoolQuery as PoolQuery, veos_lifetime_end_date as veos_lifetime_end_date, veos_lifetime_start_date as veos_lifetime_start_date
from eos.utils import decode_episode_dataframes_to_padded_arrays as decode_episode_dataframes_to_padded_arrays, decode_mongo_episodes as decode_mongo_episodes, decode_mongo_records as decode_mongo_records, dictLogger as dictLogger, logger as logger
from keras.preprocessing.sequence import pad_sequences as pad_sequences
from typing import Tuple

class MongoBuffer(Buffer[pd.DataFrame]):
    batch_size: int
    driver: Driver
    buffer_count: int
    meta: ObservationMeta
    truck: Truck
    pool: MongoPool
    query: PoolQuery
    db_config: DBConfig
    torque_table_row_names: list[str]
    logger: Incomplete
    def __post_init__(self) -> None: ...
    def load(self) -> None: ...
    def decode_batch_records(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
    def sample(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
    def find(self, idx): ...
    def close(self) -> None: ...
    def __init__(self, *, pool, batch_size, buffer_count, driver, meta, truck, query, db_config, torque_table_row_names) -> None: ...
