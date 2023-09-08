# system imports
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
from typeguard import check_type

# third-party imports
import numpy as np
import pandas as pd
import tensorflow as tf
from pymongoarrow.monkey import patch_all  # type: ignore

from eos.agent.utils.hyperparams import HyperParamRDPG
from eos.data_io.buffer import DaskBuffer, MongoBuffer  # type: ignore
from eos.data_io.struct import (
    PoolQuery,  # type: ignore
    veos_lifetime_end_date,
    veos_lifetime_start_date,
)

# local imports
from eos.utils import dictLogger, logger

from ..dpg import DPG  # type: ignore
from .actor import ActorNet  # type: ignore
from .critic import CriticNet  # type: ignore

patch_all()

"""
Title: RDPG for VEOS
Author: Binjian Xin
Date created: 2021/12/07
Last modified: 2021/12/07
Description: Adapted from DDPG


Title: Memory-based control with recurrent neural networks (RDPG)
Author: Nicolas Hees, Jonathan J Hunt, Timothy P Lillicrap, David Silver
Description: Implementing RDPG algorithm on VEOS.
"""


hyper_param_default = HyperParamRDPG()
truck_default = DPG.truck_type
actor_net_default = ActorNet(
    truck_default.observation_numel,
    truck_default.torque_flash_numel,
    hyper_param_default.HiddenDimension,  # 256
    hyper_param_default.NLayerActor,  # 2
    hyper_param_default.BatchSize,  # 4
    hyper_param_default.PaddingValue,  # -10000
    hyper_param_default.TauActor,  # 0.005
    hyper_param_default.ActorLR,  # 0.001
    Path("./actor"),
    hyper_param_default.CkptInterval,  # 5
)

actor_optimizer_default = tf.keras.optimizers.Adam(hyper_param_default.ActorLR)  # 0.001
ckpt_actor_default = tf.train.Checkpoint(
    step=tf.Variable(1),  # type: ignore
    optimizer=actor_optimizer_default,
    net=tf.keras.Model(),
)
manager_actor_default = tf.train.CheckpointManager(
    ckpt_actor_default, "./actor", max_to_keep=10
)

critic_net_default = CriticNet(
    truck_default.observation_numel,
    truck_default.torque_flash_numel,
    hyper_param_default.HiddenDimension,  # 256
    hyper_param_default.NLayerCritic,  # 2
    hyper_param_default.BatchSize,  # 4
    hyper_param_default.PaddingValue,  # -10000
    hyper_param_default.TauCritic,  # 0.005
    hyper_param_default.CriticLR,  # 0.001
    Path("./critic"),
    hyper_param_default.CkptInterval,  # 5
)
critic_optimizer_default = tf.keras.optimizers.Adam(
    hyper_param_default.CriticLR
)  # 0.002
ckpt_critic_default = tf.train.Checkpoint(
    step=tf.Variable(1),  # type: ignore
    optimizer=critic_optimizer_default,
    net=tf.keras.Model(),
)
manager_critic_default = tf.train.CheckpointManager(
    ckpt_critic_default, "./critic", max_to_keep=10
)


@dataclass
class RDPG(DPG):
    """
    RDPG agent for VEOS.
        data interface:
            - pool in mongodb
            - buffer in memory (numpy array)
        model interface:
            - actor network
            - critic network
    """

    actor_net: ActorNet = actor_net_default
    critic_net: CriticNet = critic_net_default
    target_actor_net: ActorNet = actor_net_default
    target_critic_net: CriticNet = critic_net_default
    _ckpt_actor_dir: Path = Path("")
    _ckpt_critic_dir: Path = Path("")
    logger: logging.Logger = logging.Logger("eos.agent.rdpg.rdpg")

    def __post_init__(
        self,
    ):
        """initialize the rdpg agent.

        args:
            truck.ObservationNumber (int): dimension of the state space.
            padding_value (float): value to pad the state with, impossible value for observation, action or re
        """

        self.logger = logger.getChild("main").getChild(self.__str__())
        self.logger.propagate = True
        self.dictLogger = dictLogger

        super().__post_init__()  # call DPG post_init for pool init and plot init
        self.coll_type = "EPISODE"
        self.hyper_param = HyperParamRDPG(
            HiddenDimension=256,
            PaddingValue=-10000,
            tbptt_k1=200,
            tbptt_k2=200,
            BatchSize=4,
            NStates=self.truck.observation_numel,
            NActions=self.truck.torque_flash_numel,
            ActionBias=self.truck.torque_bias,
            NLayerActor=2,
            NLayerCritic=2,
            Gamma=0.99,
            TauActor=0.005,
            TauCritic=0.005,
            ActorLR=0.001,
            CriticLR=0.001,
            CkptInterval=5,
        )

        self.buffer.query = PoolQuery(
            vehicle=self.truck.vid,
            driver=self.driver.pid,
            episodestart_start=veos_lifetime_start_date,
            episodestart_end=veos_lifetime_end_date,
            seq_len_from=10,  # from 10  # sample sequence with a length from 1 to 200
            seq_len_to=self.hyper_param.tbptt_k1 + 100,  # to 300
        )
        self.buffer.pool.query = self.buffer.query

        # actor network (w/ target network)
        self.init_checkpoint()

        self.actor_net = ActorNet(
            self.truck.observation_numel,
            self.truck.torque_flash_numel,
            self.hyper_param.HiddenDimension,  # 256
            self.hyper_param.NLayerActor,  # 2
            self.hyper_param.BatchSize,  # 4
            self.hyper_param.PaddingValue,  # -10000
            self.hyper_param.TauActor,  # 0.005
            self.hyper_param.ActorLR,  # 0.001
            self._ckpt_actor_dir,
            self.hyper_param.CkptInterval,  # 5
        )

        self.target_actor_net = ActorNet(
            self.truck.observation_numel,
            self.truck.torque_flash_numel,
            self.hyper_param.HiddenDimension,  # 256
            self.hyper_param.NLayerActor,  # 2
            self.hyper_param.BatchSize,  # 4
            self.hyper_param.PaddingValue,  # -10000
            self.hyper_param.TauActor,  # 0.005
            self.hyper_param.ActorLR,  # 0.001
            self._ckpt_actor_dir,
            self.hyper_param.CkptInterval,  # 5
        )
        # clone necessary for the first time training
        self.target_actor_net.clone_weights(self.actor_net)

        # critic network (w/ target network)

        self.critic_net = CriticNet(
            self.truck.observation_numel,
            self.truck.torque_flash_numel,
            self.hyper_param.HiddenDimension,  # 256
            self.hyper_param.NLayerCritic,  # 2
            self.hyper_param.BatchSize,  # 4
            self.hyper_param.PaddingValue,  # -10000
            self.hyper_param.TauCritic,  # 0.005
            self.hyper_param.CriticLR,  # 0.002
            self._ckpt_critic_dir,
            self.hyper_param.CkptInterval,  # 5
        )

        self.target_critic_net = CriticNet(
            self.truck.observation_numel,
            self.truck.torque_flash_numel,
            self.hyper_param.HiddenDimension,  # 256
            self.hyper_param.NLayerCritic,  # 2
            self.hyper_param.BatchSize,  # 4
            self.hyper_param.PaddingValue,  # -10000
            self.hyper_param.TauCritic,  # 0.005
            self.hyper_param.CriticLR,  # 0.002
            self._ckpt_critic_dir,
            self.hyper_param.CkptInterval,  # 5
        )
        # clone necessary for the first time training
        self.target_critic_net.clone_weights(self.critic_net)
        self.touch_gpu()

    def __repr__(self):
        return f"RDPG({self.truck.vid}, {self.driver.pid})"

    def __str__(self):
        return "RDPG"

    def __hash__(self):
        return hash(self.__repr__())

    def touch_gpu(self):
        # tf.summary.trace_on(graph=true, profiler=true)
        # ignites manual loading of tensorflow library, \
        # to guarantee the real-time processing of first data in main thread
        init_states = pd.Series(
            np.random.rand(self.truck.observation_numel)
        )  # state must have 30 (speed, throttle, current, voltage) 5 tuple

        # init_states = tf.expand_dims(input_array, 0)  # motion states is 30*2 matrix

        _ = self.actor_predict(init_states)
        self.logger.info(
            f"manual load tf library by calling convert_to_tensor",
            extra=self.dictLogger,
        )

        self.actor_net.ou_noise.reset()

        # warm up the gpu training graph execution pipeline
        if self.buffer.count() != 0:
            if not self.infer_mode:
                self.logger.info(
                    f"rdpg warm up training!",
                    extra=self.dictLogger,
                )
                (_, _) = self.train()

                self.logger.info(
                    f"rdpg warm up training done!",
                    extra=self.dictLogger,
                )

    def init_checkpoint(self):
        # actor create or restore from checkpoint
        # add checkpoints manager
        self._ckpt_actor_dir = Path(self.data_folder).joinpath(
            "tf_ckpts-"
            + self.__str__()
            + "-"
            + self.truck.vid
            + "-"
            + self.driver.pid
            + "_"
            + "/actor"
        )

        try:
            os.makedirs(self._ckpt_actor_dir)
            self.logger.info(
                "created checkpoint directory for actor: %s",
                self._ckpt_actor_dir,
                extra=self.dictLogger,
            )
        except FileExistsError:
            self.logger.info(
                "actor checkpoint directory already exists: %s",
                self._ckpt_actor_dir,
                extra=self.dictLogger,
            )

        # critic create or restore from checkpoint
        # add checkpoints manager
        self._ckpt_critic_dir = Path(self.data_folder).joinpath(
            "tf_ckpts-"
            + self.__str__()
            + "-"
            + self.truck.vid
            + "-"
            + self.driver.pid
            + "_"
            + "/critic"
        )
        try:
            os.makedirs(self._ckpt_critic_dir)
            self.logger.info(
                f"created checkpoint directory for critic: %s",
                self._ckpt_critic_dir,
                extra=self.dictLogger,
            )
        except FileExistsError:
            self.logger.info(
                f"critic checkpoint directory already exists: %s",
                self._ckpt_critic_dir,
                extra=self.dictLogger,
            )

    # TODO for infer only mode, implement a method without noisy exploration.
    def actor_predict(self, state: pd.Series) -> np.ndarray:
        """
        evaluate the actors given a single observations.
        batch size cannot be 1.
        For LSTM to be stateful, batch size must match the training scheme.
        """

        # get the current episode so far from self.observations stored by DPG.deposit()
        # self.state_t = np.ones((1, t + 1, self._num_states))
        # self.state_t[0, 0, :] = obs
        # expand the batch dimension and turn obs_t into a numpy array

        # expand states to 3D tensor [4, 1, 600] for cloud / [4, 1, 90] for kvaser
        states = tf.convert_to_tensor(
            np.expand_dims(
                np.outer(
                    np.ones((self.hyper_param.BatchSize, 1)),  # type: ignore
                    state.values,  # type: ignore
                ),
                axis=1,
            ),
            dtype=tf.float32,
        )

        # expand actions to 3D tensor [4, 1, 68] for cloud / [4, 1, 68] for kvaser
        idx = pd.IndexSlice
        try:
            last_actions = tf.convert_to_tensor(
                np.expand_dims(
                    np.outer(
                        np.ones(self.hyper_param.BatchSize),
                        self.observations[-1]  # last observation contains last action!
                        .sort_index()
                        .loc[idx["action", self.torque_table_row_names, :]]
                        .values.astype(np.float32),  # type convert to float32
                    ),
                    axis=1,  # observation (with subpart action is Multi-Indexed Series, its values are flatted
                ),  # get last_actions from last observation,
                dtype=tf.float32,  # and add batch and time dimension twice at axis 0
            )  # so that last_actions is a 3D tensor
        except (
            IndexError
        ):  # if no last action in case of the first step of the episode, then use zeros
            last_actions = tf.zeros(
                shape=(
                    self.hyper_param.BatchSize,
                    1,
                    self.truck.torque_flash_numel,
                ),  # [1, 1, 4*17]
                dtype=tf.float32,
            )  # first zero last_actions is a 3D tensor
        # self.logger.info(
        #     f"states.shape: {states.shape}; last_actions.shape: {last_actions.shape}",
        #     extra=self.dictLogger,
        # )
        # action = self.actor_net.predict(input_array)
        actions = self.actor_predict_step(
            states, last_actions
        )  # both states and last_actions are 3d tensors [B,T,D]
        action = actions.numpy()[
            0, :
        ]  # [1, 68] for cloud / [1, 68] for kvaser, squeeze the batch dimension
        # self.logger.info(f"action.shape: {action.shape}", extra=self.dictLogger)
        return action

    @tf.function(
        input_signature=[
            tf.TensorSpec(
                shape=[None, None, DPG.truck_type.observation_numel],
                dtype=tf.float32,
            ),  # [None, None, 600] for cloud / [None, None, 90] for kvaser
            tf.TensorSpec(
                shape=[None, None, DPG.truck_type.torque_flash_numel], dtype=tf.float32
            ),  # [None, None, 68] for both cloud and kvaser
        ]
    )
    def actor_predict_step(
        self, states: tf.Tensor, last_actions: tf.Tensor
    ) -> tf.Tensor:
        """
        evaluate the actors given a single observations.
        batch size is 1.
        """
        # logger.info(f"tracing", extra=self.dictLogger)
        print("tracing!")
        action = self.actor_net.predict(
            states, last_actions
        )  # already un-squeezed inside Actor function
        return action

    def train(self) -> Tuple[float, float]:
        """
        train the actor and critic moving network with truncated Backpropagation through time (TBPTT)
        with k1 = k2 = self.hyperparam.tbptt_k1 (keras)

        return:
            tuple: (actor_loss, critic_loss)
        """

        # reset the states of the stateful moving and target networks at the beginning of the training
        self.actor_net.eager_model.reset_states()
        self.critic_net.eager_model.reset_states()
        self.target_actor_net.eager_model.reset_states()
        self.target_critic_net.eager_model.reset_states()
        actor_loss = tf.constant(0.0)
        critic_loss = tf.constant(0.0)
        s_n_t, a_n_t, r_n_t, ns_n_t = self.buffer.sample()  # ignore next state for now
        # split_num = (
        #     s_n_t.shape[1] // check_type(self.hyper_param, HyperParamRDPG).tbptt_k1
        #     + 1
        #     # after padding all observations have the same length (the length of  the longest episode)
        # )  # 18//20+1=1, 50//20+1=3, for short episode if tbptt_k1> episode length, no split
        # self.logger.info(
        #     f"{{'header': 'Batch splitting', " f"'split_num': '{split_num}'}}",
        #     extra=self.dictLogger,
        # )
        # if split_num <= 0:
        #     raise ValueError("split_num <= 0, check tbptt_k1 and episode length")
        # for i, batch_t in enumerate(
        #         zip(
        #             np.array_split(
        #                 s_n_t, split_num, axis=1
        #             ),  # split on the time axis (axis=1)
        #             np.array_split(a_n_t, split_num, axis=1),
        #             np.array_split(r_n_t, split_num, axis=1),
        #             np.array_split(ns_n_t, split_num, axis=1),
        #         )
        # ):  # all actor critic have stateful LSTMs so that the LSTM states are kept between sub-batches,
        # # while trainings extend only to the end of each sub-batch by default of train_step
        # # out of tf.GradientTape() context, the tensors are detached like .detach() in pytorch

        # if s_n_t.shape[1] > check_type(self.hyper_param, HyperParamRDPG).tbptt_k1:
        ind_split = (
            np.arange(
                s_n_t.shape[1] // check_type(self.hyper_param, HyperParamRDPG).tbptt_k1
            )
            + 1
        ) * check_type(
            self.hyper_param, HyperParamRDPG
        ).tbptt_k1  # split index is np.arange(l//s)[1:]*s
        # else: for l==s, split in original array and np.array([])
        # for l<s    ind_split = np.array([])  # no split for l <= split size

        self.logger.info(
            f"{{'header': 'Batch splitting', " f"'split_num': '{ind_split}'}}",
            extra=self.dictLogger,
        )

        for i, batch_t in enumerate(  # split on the time axis (axis=1)
            zip(
                np.array_split(s_n_t, ind_split, axis=1),
                np.array_split(a_n_t, ind_split, axis=1),
                np.array_split(r_n_t, ind_split, axis=1),
                np.array_split(ns_n_t, ind_split, axis=1),
            )
        ):  # all actor critic have stateful LSTMs so that the LSTM states are kept between sub-batches,
            # while trainings extend only to the end of each sub-batch by default of train_step
            # out of tf.GradientTape() context, the tensors are detached like .detach() in pytorch
            s_n_t_sub, a_n_t_sub, r_n_t_sub, ns_n_t_sub = batch_t
            if s_n_t_sub is np.array([]):
                self.logger.warning(
                    f"batch sub sequence s_n_t: {i} is empty!", extra=self.dictLogger
                )
                continue
            else:
                self.logger.info(
                    f"batch sub sequences s_n_t: {i} is valid. ", extra=self.dictLogger
                )

            actor_loss, critic_loss = self.train_step(
                s_n_t_sub, a_n_t_sub, r_n_t_sub, ns_n_t_sub
            )
            self.logger.info(
                f"batch actor loss: {actor_loss.numpy()}; batch critic loss: {critic_loss.numpy()}",
                extra=self.dictLogger,
            )

        # return the last actor and critic loss
        # return actor_loss.numpy()[0], critic_loss.numpy()[0]
        return actor_loss.numpy(), critic_loss.numpy()

    @tf.function(
        input_signature=[
            tf.TensorSpec(
                shape=[
                    DPG.rdpg_hyper_type.BatchSize,
                    None,
                    DPG.truck_type.observation_numel,
                ],
                dtype=tf.float32,
            ),
            tf.TensorSpec(
                shape=[
                    DPG.rdpg_hyper_type.BatchSize,
                    None,
                    DPG.truck_type.torque_flash_numel,
                ],
                dtype=tf.float32,
            ),
            tf.TensorSpec(
                shape=[DPG.rdpg_hyper_type.BatchSize, None, 1], dtype=tf.float32
            ),
            tf.TensorSpec(
                shape=[
                    DPG.rdpg_hyper_type.BatchSize,
                    None,
                    DPG.truck_type.observation_numel,
                ],
                dtype=tf.float32,
            ),
        ]
    )
    def train_step(self, s_n_t, a_n_t, r_n_t, ns_n_t) -> Tuple[tf.Tensor, tf.Tensor]:
        # train critic using bptt
        print("tracing train_step!")
        self.logger.info(f"start train_step with tracing")
        # logger.info(f"start train_step")

        gamma = tf.convert_to_tensor(self.hyper_param.Gamma, dtype=tf.float32)

        with tf.GradientTape() as tape:
            # actions at h_t+1
            self.logger.info(f"start evaluate_actions")
            t_a_ht1 = self.target_actor_net.evaluate_actions(
                ns_n_t[:, 1:, :],  # s_1, s_2, ..., s_n, ...
                a_n_t[:, 1:, :],  # a_0, a_1, ..., a_{n-1}, ...
            )  # t_actor(h_{t+1}): [h_1(a_0, s_1), h_2(a_1, s_2), ..., h_n(a_{n-1}, s_n), ...]
            print(f"t_a_ht1.shape: {t_a_ht1.shape}, type: {t_a_ht1.dtype}")

            # state action value at h_t+1
            # logger.info(f"o_n_t.shape: {self.o_n_t.shape}")
            # logger.info(f"t_a_ht1.shape: {self.t_a_ht1.shape}")
            self.logger.info(f"start critic evaluate_q")
            t_q_ht1 = self.target_critic_net.evaluate_q(
                ns_n_t[:, 1:, :], a_n_t[:, 1:, :], t_a_ht1
            )  # t_critic(h_{t+1}, t_actor(h_{t+1})): [h_1(s_1, a_0), h_2(s_2, a_1), ..., h_n(s_n, a_{n-1}), ...]
            # self.logger.info(f"critic evaluate_q done, t_q_ht1.shape: {t_q_ht1.shape}")
            print(
                f"critic evaluate_q done, t_q_ht1.shape: {t_q_ht1[:, :-1,:].shape}, type: {t_q_ht1.dtype}"
            )
            print(f"r_n_t.shape: {r_n_t[:,1:,:].shape}, type: {r_n_t.dtype}")

            # logger.info(f"t_q_ht_bl.shape: {t_q_ht_bl.shape}")
            # y_n_t shape (batch_size, seq_len, 1), target value
            y_n_t = (
                r_n_t[:, 1:, :] + gamma * t_q_ht1  # fix t_q_ht1[:, :-1, :]!
            )  # y0(r_0, Q(h_1,mu(h_1))), y1(r_1, Q(h_2,mu(h_2)), ...)
            # self.logger.info(f"y_n_t.shape: {y_n_t.shape}")
            print(f"y_n_t.shape: {y_n_t.shape}")

            # scalar value, average over the batch, time steps
            critic_loss = tf.math.reduce_mean(
                y_n_t
                - self.critic_net.evaluate_q(
                    s_n_t[:, 1:, :], a_n_t[:, :-1, :], a_n_t[:, 1:, :]
                )  # Q(s_t, a_{t-1}, a_t): Q(s_0, a_-1, a_0), Q(s_1, a_0, a_1), ..., Q(s_n, a_{n-1}, a_n)
            )
        critic_grad = tape.gradient(
            critic_loss, self.critic_net.eager_model.trainable_variables
        )
        self.critic_net.optimizer.apply_gradients(
            zip(critic_grad, self.critic_net.eager_model.trainable_variables)
        )
        # self.logger.info(f"applied critic gradient", extra=dictLogger)
        print(f"applied critic gradient")

        # train actor using bptt
        with tf.GradientTape() as tape:
            # self.logger.info(f"start actor evaluate_actions", extra=dictLogger)
            print(f"start actor evaluate_actions")
            a_ht = self.actor_net.evaluate_actions(
                s_n_t[:, 1:, :], a_n_t[:, :-1, :]
            )  # states, last actions: (s_0, a_-1), (s_1, a_0), ..., (s_n, a_{n-1})
            # self.logger.info(
            #     f"actor evaluate_actions done, a_ht.shape: {a_ht.shape}",
            #     extra=dictLogger,
            # )
            print(f"actor evaluate_actions done, a_ht.shape: {a_ht.shape}")
            q_ht = self.critic_net.evaluate_q(s_n_t[:, 1:, :], a_n_t[:, :-1, :], a_ht)
            # self.logger.info(
            #     f"actor evaluate_q done, q_ht.shape: {q_ht.shape}",
            #     extra=dictLogger,
            # )
            print(f"actor evaluate_q done, q_ht.shape: {q_ht.shape}")
            # logger.info(f"a_ht.shape: {self.a_ht.shape}")
            # logger.info(f"q_ht.shape: {self.q_ht.shape}")
            # -1 because we want to maximize the q_ht
            # scalar value, average over the batch and time steps
            actor_loss = -tf.math.reduce_mean(q_ht)

        actor_grad = tape.gradient(
            actor_loss, self.actor_net.eager_model.trainable_variables
        )
        # logger.info(f"action_gradients: {action_gradients}")
        # logger.info(f"actor_grad_weight: {actor_grad_weight} vs actor_grad: {actor_grad}")
        # logger.info(f"the grad diff: {actor_grad - actor_grad_weight}")
        self.actor_net.optimizer.apply_gradients(
            zip(actor_grad, self.actor_net.eager_model.trainable_variables)
        )
        # self.logger.info(f"applied actor gradient", extra=dictLogger)
        print(f"applied actor gradient")

        return actor_loss, critic_loss

    def end_episode(self):
        super().end_episode()
        # reset the states of the actor and critic networks
        self.actor_net.eager_model.reset_states()
        self.critic_net.eager_model.reset_states()
        self.target_actor_net.eager_model.reset_states()
        self.target_critic_net.eager_model.reset_states()

    def get_losses(self):
        pass

    def notrain(self):
        """
        purely evaluate the actor and critic networks to  return the losses without training.

        return:
            tuple: (actor_loss, critic_loss)
        """

        s_n_t, a_n_t, r_n_t, ns_n_t = self.buffer.sample()  # ignore the next state

        # get critic loss
        # actions at h_t+1
        t_a_ht1 = self.target_actor_net.evaluate_actions(
            ns_n_t[:, 1:, :],  # s_1, s_2, ..., s_n
            a_n_t[:, 1:, :],  # a_0, a_1, ..., a_{n-1}
        )  # t_actor(h_{t+1}, s_{t+1}): [h_1(s_1), h_2(s_2), ..., h_n(s_n), ...]

        # state action value at h_t+1
        t_q_ht1 = self.target_critic_net.evaluate_q(
            ns_n_t[:, 1:, :], a_n_t[:, 1:, :], t_a_ht1
        )  # t_critic(h_{t+1}, t_actor(h_{t+1})): [h_1(s_1, a_0), h_2(s_2, a_1), ..., h_n(s_n, a_{n-1}), ...]

        # y_n_t shape (batch_size, seq_len, 1)
        y_n_t = (
            r_n_t[:, 1:, :] + self.hyper_param.Gamma * t_q_ht1
        )  # y0(r_0, Q(h_1,mu(h_1))), y1(r_1, Q(h_2, mu(h_2)), ...)

        # scalar value, average over the batch, time steps
        critic_loss = tf.math.reduce_mean(
            y_n_t
            - self.critic_net.evaluate_q(
                s_n_t[:, 1:, :], a_n_t[:, :-1, :], a_n_t[:, 1:, :]
            )
        )  # Q(s_t, a_{t-1}, a_t): Q(s_0, a_-1, a_0), Q(s_1, a_0, a_1), ..., Q(s_n, a_{n-1}, a_n)

        # get  actor loss
        a_ht = self.actor_net.evaluate_actions(
            s_n_t[:, 1:, :], a_n_t[:, :-1, :]
        )  # states, last actions: (s_0, a_-1), (s_1, a_0), ..., (s_n, a_{n-1})
        q_ht = self.critic_net.evaluate_q(s_n_t[:, 1:, :], a_n_t[:, :-1, :], a_ht)

        # -1 because we want to maximize the q_ht
        # scalar value, average over the batch and time steps
        actor_loss = tf.math.reduce_mean(-q_ht)

        return actor_loss, critic_loss

    def soft_update_target(self):
        """
        update target networks with tiny tau value, typical value 0.001.
        done after each batch, slowly update target by polyak averaging.
        """
        self.target_critic_net.soft_update(self.critic_net)
        self.target_actor_net.soft_update(self.actor_net)

    def save_ckpt(self):
        self.actor_net.save_ckpt()
        self.critic_net.save_ckpt()
