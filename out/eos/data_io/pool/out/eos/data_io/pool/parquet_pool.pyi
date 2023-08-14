import dask.dataframe as dd
import pandas as pd
from _typeshed import Incomplete as Incomplete
from eos.data_io.pool.dask_pool import DaskPool as DaskPool
from eos.data_io.struct import PoolQuery as PoolQuery
from typing import Optional

class ParquetPool(DaskPool):
    ddf: dd.DataFrame
    ddf_list: list[dd.DataFrame]
    logger: Incomplete
    dictLogger: Incomplete
    def __post_init__(self) -> None: ...
    input_metadata: Incomplete
    bEmpty: bool
    def load(self) -> None: ...
    def close(self) -> None: ...
    def store(self, episode: pd.DataFrame) -> None: ...
    def delete(self, idx) -> None: ...
    def delete_episode(self, query: PoolQuery) -> None: ...
    def get_query(self, query: Optional[PoolQuery] = ...) -> dd.DataFrame: ...
    def sample(self, size: int = ..., *, query: PoolQuery) -> pd.DataFrame: ...
    def __iter__(self): ...
    def __init__(self, *, recipe, query, meta, pl_path, input_metadata, bEmpty, ddf, ddf_list) -> None: ...
