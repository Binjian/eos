import abc
from typing import Any, Generic, Optional, Union

import pandas as pd

from eos.data_io.struct import ItemT as ItemT
from eos.data_io.struct import PoolQuery as PoolQuery

class Pool(abc.ABC, Generic[ItemT], metaclass=abc.ABCMeta):
    def __init_subclass__(cls) -> None: ...
    def __post_init__(self) -> None: ...
    @abc.abstractmethod
    def load(self): ...
    @abc.abstractmethod
    def close(self): ...
    @abc.abstractmethod
    def store(self, item: DocItemT): ...
    @abc.abstractmethod
    def delete(self, idx): ...
    @abc.abstractmethod
    def count(self, query: Optional[PoolQuery] = ...) -> int: ...
    @abc.abstractmethod
    def find(self, query: PoolQuery) -> Any: ...
    @abc.abstractmethod
    def sample(self, size: int, *, query: Optional[PoolQuery] = ...) -> Union[pd.DataFrame, list[DocItemT]]: ...
    @abc.abstractmethod
    def __iter__(self) -> Any: ...
    def __getitem__(self, query: PoolQuery) -> Any: ...
    def __len__(self) -> int: ...
