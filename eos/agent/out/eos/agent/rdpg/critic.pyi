from _typeshed import Incomplete

from eos.utils import dictLogger as dictLogger
from eos.utils import logger as logger
from eos.utils.exception import ReadOnlyError as ReadOnlyError

class CriticNet:
    eager_model: Incomplete
    optimizer: Incomplete
    ckpt_dir: Incomplete
    ckpt: Incomplete
    ckpt_manager: Incomplete
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim,
        n_layers,
        padding_value,
        tau,
        lr,
        ckpt_dir,
        ckpt_interval,
    ) -> None: ...
    def clone_weights(self, moving_net) -> None: ...
    def soft_update(self, moving_net) -> None: ...
    def save_ckpt(self) -> None: ...
    def evaluate_q(self, state, action): ...
    @property
    def state_dim(self): ...
    @property
    def action_dim(self): ...
    @property
    def hidden_dim(self): ...
    @property
    def lr(self): ...
    @property
    def padding_value(self): ...
    @property
    def n_layers(self): ...
    @property
    def tau(self): ...
    @property
    def ckpt_interval(self): ...
