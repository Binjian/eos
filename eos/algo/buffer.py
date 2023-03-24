from __future__ import annotations
import abc
from dataclasses import dataclass
from typing import Optional, Any
import weakref


@dataclass
class Buffer(abc.ABC):
    """
    Buffer is the internal dynamic memory object for pooling the experience tuples.
    It can have PoolMixin as storage in mongodb or numpy array file.
    It can provide data wrapper for the experience tuples or episode tuples
    and multi-inheritance for the pool.
    It can provide load(), save(), store(), sample()
    """

    def __post_init__(self):
        """User weakref finalizer to make sure close is called when the object is destroyed"""
        self._finalizer = weakref.finalize(self, self.close)

    @abc.abstractmethod
    def load(self):
        """
        load buffer from pool
        """
        pass

    @abc.abstractmethod
    def close(self):
        """
        close the pool, for destructor
        """
        pass

    @abc.abstractmethod
    def store(self, item: dict):
        """
        Deposit an item (record/episode) into the pool
        """
        pass

    @abc.abstractmethod
    def find(self, idx):
        """
        find an itme by id or name.
        """
        pass

    @abc.abstractmethod
    def sample(self) -> tuple:
        """
        Update the actor and critic networks using the sampled batch.
        """
        pass

    @abc.abstractmethod
    def count(self, query: Optional[dict] = None):
        """
        Count the number of records in the db.
        rule is an optional dictionary specifying a rule or
        a pipeline in mongodb
        query = {
            vehicle_id: str = "VB7",
            driver_id: str = "longfei-zheng",
            dt_start: datetime = None,
            dt_end: datetime = None,
        }
        """
        pass
