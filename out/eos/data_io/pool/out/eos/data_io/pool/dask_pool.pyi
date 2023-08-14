import abc
import dask.bag as db
import dask.dataframe as dd
import pandas as pd
from _typeshed import Incomplete as Incomplete
from configparser import ConfigParser
from eos.data_io.pool import Pool as Pool
from eos.data_io.struct import ObservationMeta as ObservationMeta, PoolQuery as PoolQuery
from pathlib import Path
from typing import Dict, Optional, Union

class DaskPool(Pool[pd.DataFrame], metaclass=abc.ABCMeta):
    recipe: ConfigParser
    query: PoolQuery
    meta: ObservationMeta
    pl_path: Path
    input_metadata: Dict
    bEmpty: bool
    array_names: Incomplete
    def __post_init__(self) -> None: ...
    def find(self, query: PoolQuery) -> Optional[pd.DataFrame]: ...
    @abc.abstractmethod
    def get_query(self, query: Optional[PoolQuery] = ...) -> Union[dd.DataFrame, db.Bag]: ...
    def count(self, query: Optional[PoolQuery] = ...): ...
    @abc.abstractmethod
    def sample(self, size: int, *, query: Optional[PoolQuery] = ...) -> pd.DataFrame: ...
    def __init__(self, *, recipe, query, meta, pl_path, input_metadata, bEmpty) -> None: ...
