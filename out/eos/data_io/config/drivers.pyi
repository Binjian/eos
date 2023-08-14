import pandas as pd
from _typeshed import Incomplete
from ordered_set import OrderedSet
from typing import Optional

DriverCat: Incomplete
RE_DRIVER: str

class Driver:
    pid: str
    name: str
    site: str
    contract_range: Optional[pd.date_range]
    cat: OrderedSet
    def __post_init__(self) -> None: ...
    def __init__(self, pid, name, site, contract_range, cat) -> None: ...

drivers: Incomplete
drivers_by_id: Incomplete
