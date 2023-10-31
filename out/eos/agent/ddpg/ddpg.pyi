import logging
from pathlib import Path

import pandas as pd
import tensorflow as tf
from _typeshed import Incomplete

from eos.agent.dpg import DPG as DPG
from eos.agent.utils import HYPER_PARAM as HYPER_PARAM
from eos.agent.utils import OUActionNoise as OUActionNoise
from eos.agent.utils import hyper_param_by_name as hyper_param_by_name
from eos.data_io.buffer import DaskBuffer as DaskBuffer
from eos.data_io.buffer import MongoBuffer as MongoBuffer
from eos.utils import dictLogger as dictLogger
from eos.utils import logger as logger

class DDPG(DPG):
    logger: logging.Logger
    manager_critic: tf.train.CheckpointManager
    ckpt_critic: tf.train.Checkpoint
    manager_actor: tf.train.CheckpointManager
    ckpt_actor: tf.train.Checkpoint
    actor_saved_model_path: Path
    critic_saved_model_path: Path
    dictLogger: Incomplete
    coll_type: str
    hyper_param: Incomplete
    actor_optimizer: Incomplete
    critic_optimizer: Incomplete
    ou_noise_std_dev: float
    ou_noise: Incomplete
    def __post_init__(self) -> None: ...
    def __hash__(self): ...
    def init_checkpoint(self) -> None: ...
    def save_as_saved_model(self) -> None: ...
    def load_saved_model(self): ...
    def convert_to_tflite(self) -> None: ...
    @classmethod
    def model_summary_print(cls, mdl: tf.keras.Model, file_path: Path): ...
    @classmethod
    def tflite_analytics_print(cls, tflite_file_path: Path): ...
    def save_ckpt(self) -> None: ...
    def update_target(self, target_weights, weights, tau) -> None: ...
    def soft_update_target(self) -> None: ...
    @classmethod
    def get_actor(
        cls,
        num_states: int,
        num_actions: int,
        num_hidden: int = ...,
        num_layers: int = ...,
        action_bias: float = ...,
    ): ...
    @classmethod
    def get_critic(
        cls,
        num_states: int,
        num_actions: int,
        num_hidden0: int = ...,
        num_hidden1: int = ...,
        num_hidden2: int = ...,
        num_layers: int = ...,
    ): ...
    def policy(self, state: pd.Series): ...
    def actor_predict(self, state: pd.Series, t: int): ...
    def infer_single_sample(self, state_flat: tf.Tensor): ...
    def touch_gpu(self) -> None: ...
    def sample_minibatch(self): ...
    def train(self): ...
    def update_with_batch(
        self,
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
        training: bool = ...,
    ): ...
    def get_losses(self): ...
    @property
    def actor_model(self) -> tf.keras.Model: ...
    @property
    def critic_model(self) -> tf.keras.Model: ...
    @property
    def target_actor_model(self) -> tf.keras.Model: ...
    @property
    def target_critic_model(self) -> tf.keras.Model: ...
    def __init__(
        self,
        _buffer,
        _hyper_param,
        _episode_start_dt,
        logger,
        _actor_model,
        _critic_model,
        _target_actor_model,
        _target_critic_model,
        manager_critic,
        ckpt_critic,
        manager_actor,
        ckpt_actor,
        actor_saved_model_path,
        critic_saved_model_path,
        *,
        _truck,
        _driver,
        _coll_type,
        _pool_key,
        _data_folder,
        _infer_mode,
        _observation_meta,
        _resume,
        _observations,
        _torque_table_row_names,
        _epi_no
    ) -> None: ...
