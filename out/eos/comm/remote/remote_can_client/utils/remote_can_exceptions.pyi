from collections import UserDict
from dataclasses import field as field
from typing import Optional

from requests.exceptions import ConnectionError as ConnectionError
from requests.exceptions import HTTPError as HTTPError
from requests.exceptions import InvalidJSONError as InvalidJSONError
from requests.exceptions import ReadTimeout as ReadTimeout
from requests.exceptions import RequestException as RequestException
from requests.exceptions import Timeout as Timeout

class RemoteCanException(Exception):
    err_code: Optional[int]
    extra_msg: Optional[str]
    codes: UserDict
    def __init__(self, *, err_code, extra_msg, codes) -> None: ...
