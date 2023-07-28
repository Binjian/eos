import abc
import numpy as np
import pandas as pd
from eos.data_io.pool import Pool as Pool
from eos.data_io.struct import DocItemT as DocItemT, PoolQuery as PoolQuery
from typing import Generic, Optional, Tuple

class Buffer(abc.ABC, Generic[DocItemT], metaclass=abc.ABCMeta):
    pool: Pool
    batch_size: int
    buffer_count: int
    def __init_subclass__(cls) -> None: ...
    def __post_init__(self) -> None: ...
    @abc.abstractmethod
    def load(self): ...
    @abc.abstractmethod
    def close(self): ...
    def store(self, episode: pd.DataFrame): ...
    def find(self, idx): ...
    @abc.abstractmethod
    def sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
    def count(self, query: Optional[PoolQuery] = ...): ...
    def __init__(self, *, pool, batch_size, buffer_count) -> None: ...
