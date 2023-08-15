from typing import Optional, Union

import pandas as pd
from _typeshed import Incomplete
from bson.codec_options import CodecOptions
from pymongo import MongoClient
from pymongo.collection import Collection as Collection
from pymongo.database import Database as Database

from eos.data_io.config import DBConfig as DBConfig
from eos.data_io.config import \
    db_config_servers_by_name as db_config_servers_by_name
from eos.data_io.pool import Pool as Pool
from eos.data_io.struct import DataFrameDoc as DataFrameDoc
from eos.data_io.struct import ObservationMeta as ObservationMeta
from eos.data_io.struct import PoolQuery as PoolQuery
from eos.data_io.struct import \
    veos_lifetime_start_date as veos_lifetime_start_date
from eos.utils import dictLogger as dictLogger
from eos.utils import eos_df_to_nested_dict as eos_df_to_nested_dict
from eos.utils import logger as logger

class MongoPool(Pool[pd.DataFrame]):
    client: MongoClient
    db: Database
    coll_name: str
    collection: Collection[DataFrameDoc]
    meta: ObservationMeta
    db_config: DBConfig
    query: PoolQuery
    codec_option: CodecOptions
    logger: Incomplete
    dictLogger: Incomplete
    def __post_init__(self) -> None: ...
    def load(self) -> None: ...
    def close(self) -> None: ...
    def drop_collection(self) -> None: ...
    def parse_query(self, query: Union[PoolQuery | PoolQuery]) -> dict: ...
    def store_record(self, episode: pd.DataFrame): ...
    def store_episode(self, episode: pd.DataFrame): ...
    def store(self, episode: pd.DataFrame): ...
    def find(self, query: Union[PoolQuery | PoolQuery]) -> Optional[pd.DataFrame]: ...
    def delete(self, item_id): ...
    def __iter__(self): ...
    def count(self, query: Optional[PoolQuery | PoolQuery] = ...): ...
    def sample(self, size: int = ..., *, query: Optional[PoolQuery] = ...) -> pd.DataFrame: ...
    def __init__(self, *, client, db, coll_name, collection, meta, db_config, query, codec_option) -> None: ...
