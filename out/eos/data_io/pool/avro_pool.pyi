from typing import Optional

import dask.bag as db
import pandas as pd
from _typeshed import Incomplete
from dask.bag import Bag as Bag
from dask.bag import random as random

from eos.data_io.pool.dask_pool import DaskPool as DaskPool
from eos.data_io.pool.episode_avro_schema import \
    gen_episode_schema as gen_episode_schema
from eos.data_io.struct import ObservationMeta as ObservationMeta
from eos.data_io.struct import PoolQuery as PoolQuery
from eos.data_io.struct import veos_lifetime_end_date as veos_lifetime_end_date
from eos.data_io.struct import \
    veos_lifetime_start_date as veos_lifetime_start_date
from eos.utils import avro_ep_decoding as avro_ep_decoding
from eos.utils import avro_ep_encoding as avro_ep_encoding
from eos.utils import dictLogger as dictLogger
from eos.utils import logger as logger

class AvroPool(DaskPool):
    dbg: db.Bag
    dbg_schema: dict
    logger: Incomplete
    dictLogger: Incomplete
    def __post_init__(self) -> None: ...
    input_metadata: Incomplete
    bEmpty: bool
    def load(self): ...
    def close(self) -> None: ...
    def store(self, episode: pd.DataFrame) -> None: ...
    def get_query(self, query: Optional[PoolQuery] = ...) -> Bag: ...
    def find(self, query: PoolQuery) -> Optional[pd.DataFrame]: ...
    def delete(self, idx) -> None: ...
    def remove_episode(self, query: PoolQuery) -> None: ...
    def sample(self, size: int = ..., *, query: Optional[PoolQuery] = ...) -> pd.DataFrame: ...
    def __iter__(self): ...
    def __init__(self, *, recipe, query, meta, pl_path, input_metadata, bEmpty, dbg, dbg_schema) -> None: ...
