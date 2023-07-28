from collections import UserDict
from dataclasses import field as field
from requests.exceptions import ConnectionError as ConnectionError, HTTPError as HTTPError, InvalidJSONError as InvalidJSONError, ReadTimeout as ReadTimeout, RequestException as RequestException, Timeout as Timeout
from typing import Optional

class RemoteCanException(Exception):
    err_code: Optional[int]
    extra_msg: Optional[str]
    codes: UserDict
    def __init__(self, *, err_code, extra_msg, codes) -> None: ...
