from datetime import datetime

from _typeshed import Incomplete
from bson import ObjectId as ObjectId
from pymongoarrow.api import Schema as Schema

from eos.utils import dictLogger as dictLogger
from eos.utils import logger as logger

class Pool:
    logger: Incomplete
    dictLogger: Incomplete
    debug: Incomplete
    db_name: Incomplete
    coll_name: Incomplete
    schema: Incomplete
    client: Incomplete
    db: Incomplete
    def __init__(self, url: str = ..., username: str = ..., password: str = ..., db_name: str = ..., coll_name: str = ..., schema: dict = ..., debug: bool = ...) -> None: ...
    def drop_mongo(self) -> None: ...
    def drop_collection(self) -> None: ...
    def deposit_item(self, record): ...
    def count_items(self, vehicle_id: str = ..., driver_id: str = ..., dt_start: datetime = ..., dt_end: datetime = ...): ...
    def find_item(self, id): ...
    def sample_batch_items(self, batch_size: int = ..., vehicle_id: str = ..., driver_id: str = ..., dt_start: datetime = ..., dt_end: datetime = ...): ...
