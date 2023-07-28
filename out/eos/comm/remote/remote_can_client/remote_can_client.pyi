import pandas as pd
from .utils.remote_can_exceptions import RemoteCanException as RemoteCanException
from _typeshed import Incomplete
from eos.utils import dictLogger as dictLogger, logger as logger
from requests.adapters import HTTPAdapter
from typing import Dict, Optional

DEFAULT_TIMEOUT: int

class TimeoutHTTPAdapter(HTTPAdapter):
    timeout: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def send(self, request, **kwargs): ...

class RemoteCanClient:
    logger: Incomplete
    dictLogger: Incomplete
    url: Incomplete
    proxies: Incomplete
    session: Incomplete
    retries: Incomplete
    truckname: Incomplete
    truck: Incomplete
    def __init__(self, url: str = ..., proxies: Optional[Dict] = ..., truckname: str = ...) -> None: ...
    def send_torque_map(self, pedalmap: pd.DataFrame, swap: bool = ..., timeout: int = ...): ...
    def get_signals(self, duration, timeout: int = ...): ...
