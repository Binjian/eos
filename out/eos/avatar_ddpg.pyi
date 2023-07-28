from _typeshed import Incomplete
from agent.utils import HYPER_PARAM as HYPER_PARAM
from avatar import Avatar
from eos import projroot as projroot
from eos.utils import dictLogger as dictLogger, logger as logger

class AvatarDDPG(Avatar):
    hyper_param: HYPER_PARAM
    agent: Incomplete
    def __post__init__(self) -> None: ...
    def __init__(self, *, hyper_param, **) -> None: ...
