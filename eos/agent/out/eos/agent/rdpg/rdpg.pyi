import logging
import pandas as pd
from ..dpg import DPG as DPG
from ..hyperparams import (
    HyperParam as HyperParam,
    hyper_param_by_name as hyper_param_by_name,
)
from .actor import ActorNet as ActorNet
from .critic import CriticNet as CriticNet
from _typeshed import Incomplete
from eos.data_io.buffer import DaskBuffer as DaskBuffer, MongoBuffer as MongoBuffer
from eos.data_io.struct import EpisodeDoc as EpisodeDoc
from eos.utils import dictLogger as dictLogger, logger as logger
from keras.utils import pad_sequences as pad_sequences

class RDPG(DPG):
    logger: logging.Logger
    actor_net: ActorNet
    critic_net: CriticNet
    target_actor_net: ActorNet
    target_critic_net: CriticNet
    state_t: list
    R: list
    h_t: list
    buffer_count: int
    dictLogger: Incomplete
    coll_type: str
    hyper_param: Incomplete
    def __post_init__(self) -> None: ...
    def touch_gpu(self) -> None: ...
    def init_checkpoint(self) -> None: ...
    def actor_predict(self, state: pd.Series, t: int): ...
    def actor_predict_step(self, obs): ...
    def train(self): ...
    def train_step(self, s_n_t, a_n_t, r_n_t): ...
    def get_losses(self) -> None: ...
    def notrain(self): ...
    def soft_update_target(self) -> None: ...
    def save_ckpt(self) -> None: ...
    def __init__(
        self,
        logger,
        actor_net,
        critic_net,
        target_actor_net,
        target_critic_net,
        state_t,
        R,
        h_t,
        buffer_count,
        _seq_len,
        _ckpt_actor_dir,
        _ckpt_critic_dir,
        *,
        _truck,
        _driver,
        _buffer,
        _coll_type,
        _hyper_param,
        _pool_key,
        _data_folder,
        _infer_mode,
        _observation_meta,
        _episode_start_dt,
        _resume,
        _observations,
        _torque_table_row_names,
        _epi_no
    ) -> None: ...
