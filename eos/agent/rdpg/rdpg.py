# system imports
import os
from dataclasses import dataclass
import logging
from pathlib import Path

# third-party imports
import numpy as np
import pandas as pd
import tensorflow as tf

from keras.utils import pad_sequences  # type: ignore
from pymongoarrow.monkey import patch_all  # type: ignore

# local imports
from eos.utils import dictLogger, logger
from ..dpg import DPG  # type: ignore
from eos.agent.utils.hyperparams import HyperParamRDPG

from .actor import ActorNet  # type: ignore
from .critic import CriticNet  # type: ignore

from eos.data_io.buffer import MongoBuffer, DaskBuffer  # type: ignore

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

    logger: logging.Logger = logging.Logger('eos.agent.rdpg.rdpg')
    actor_net: ActorNet = ActorNet()
    critic_net: CriticNet = CriticNet()
    target_actor_net: ActorNet = ActorNet()
    target_critic_net: CriticNet = CriticNet()
    _ckpt_actor_dir: Path = Path('')
    _ckpt_critic_dir: Path = Path('')

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
        self.hyper_param = HyperParamRDPG("RDPG")

        # actor network (w/ target network)
        self.init_checkpoint()

        self.actor_net = ActorNet(
            self.truck.observation_numel,
            self.truck.torque_flash_numel,
            self.hyper_param.HiddenDimension,  # 256
            self.hyper_param.NLayerActor,  # 2
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
        return f"RDPG({self.truck.name}, {self.driver})"

    def __str__(self):
        return "RDPG"

    def touch_gpu(self):
        # tf.summary.trace_on(graph=true, profiler=true)
        # ignites manual loading of tensorflow library, \
        # to guarantee the real-time processing of first data in main thread
        init_motion_power = np.random.rand(self.truck.observation_numel)
        init_states = tf.convert_to_tensor(
            init_motion_power
        )  # state must have 30 (speed, throttle, current, voltage) 5 tuple
        input_array = tf.cast(init_states, dtype=tf.float32)

        # init_states = tf.expand_dims(input_array, 0)  # motion states is 30*2 matrix

        _ = self.actor_predict(input_array)
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
        self._ckpt_actor_dir = (
            self.data_folder
            + "-"
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
        self._ckpt_critic_dir = (
            self.data_folder
            + "-"
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
    def actor_predict(self, state: pd.Series):
        """
        evaluate the actors given a single observations.
        batchsize is 1.
        """

        # get the current episode so far from self.observations stored by DPG.deposit()
        # self.state_t = np.ones((1, t + 1, self._num_states))
        # self.state_t[0, 0, :] = obs
        # expand the batch dimension and turn obs_t into a numpy array

        states = tf.expand_dims(
            tf.expand_dims(
                tf.convert_to_tensor(state.values),
                axis=0,  # state is Multi-Indexed Series, its values are flatted
            ),
        )  # add batch and time dimension twice at axis 0, so that states is a 3D tensor
        idx = pd.IndexSlice
        try:
            last_actions = tf.expand_dims(
                tf.expand_dims(
                    self.observations[-1]  # last observation contains last action!
                    .sort_index()
                    .loc[idx['action', self.torque_table_row_names, :]]
                    .values.astype(np.float32),  # type convert to float32
                    axis=0,  # observation (with subpart action is Multi-Indexed Series, its values are flatted
                ),  # get last_actions from last observation,
                axis=0,  # and add batch and time dimension twice at axis 0
            )  # so that last_actions is a 3D tensor
        except (
            IndexError
        ):  # if no last action in case of the first step of the episode, then use zeros
            last_actions = tf.zeros(
                shape=(1, 1, self.truck.torque_flash_numel),  # [1, 1, 4*17]
                dtype=tf.float32,
            )  # first zero last_actions is a 3D tensor
        self.logger.info(
            f"states.shape: {states.shape}; last_actions.shape: {last_actions.shape}",
            extra=self.dictLogger,
        )
        # action = self.actor_net.predict(input_array)
        action = self.actor_predict_step(
            states, last_actions
        )  # both states and last_actions are 3d tensors [B,T,D]
        self.logger.info(f"action.shape: {action.shape}", extra=self.dictLogger)
        return action

    @tf.function(
        input_signature=[
            tf.TensorSpec(
                shape=[None, None, DPG._truck_type.observation_numel],
                dtype=tf.float32,
            ),  # [None, None, 600] for cloud / [None, None, 90] for kvaser
            tf.TensorSpec(
                shape=[None, None, DPG._truck_type.torque_flash_numel], dtype=tf.float32
            ),  # [None, None, 68] for both cloud and kvaser
        ]
    )
    def actor_predict_step(
        self, states: tf.Tensor, last_actions: tf.Tensor
    ) -> tf.Tensor:
        """
        evaluate the actors given a single observations.
        batchsize is 1.
        """
        # logger.info(f"tracing", extra=self.dictLogger)
        print("tracing!")
        action = self.actor_net.predict(
            states, last_actions
        )  # already un-squeezed inside Actor function
        return action

    def train(self):
        """
        train the actor and critic moving network.

        return:
            tuple: (actor_loss, critic_loss)
        """

        s_n_t, a_n_t, r_n_t, ns_n_t = self.buffer.sample()  # ignore next state for now
        actor_loss, critic_loss = self.train_step(s_n_t, a_n_t, r_n_t, ns_n_t)
        return actor_loss, critic_loss

    # @tf.function(input_signature=[tf.tensorspec(shape=[none,none,1], dtype=tf.float32),
    #                               tf.tensorspec(shape=[none,none,90], dtype=tf.float32),
    #                               tf.tensorspec(shape=[none,none,85], dtype=tf.float32)])
    def train_step(self, s_n_t, a_n_t, r_n_t, ns_n_t):
        # train critic using bptt
        print("tracing train_step!")
        self.logger.info(f"start train_step with tracing")
        # logger.info(f"start train_step")
        with tf.GradientTape() as tape:
            # actions at h_t+1
            self.logger.info(f"start evaluate_actions")
            t_a_ht1 = self.target_actor_net.evaluate_actions(
                ns_n_t[:, 1:, :],  # s_1, s_2, ..., s_n, ...
                a_n_t[:, 1:, :],  # a_0, a_1, ..., a_{n-1}, ...
            )  # t_actor(h_{t+1}): [h_1(a_0, s_1), h_2(a_1, s_2), ..., h_n(a_{n-1}, s_n), ...]

            # state action value at h_t+1
            # logger.info(f"o_n_t.shape: {self.o_n_t.shape}")
            # logger.info(f"t_a_ht1.shape: {self.t_a_ht1.shape}")
            self.logger.info(f"start critic evaluate_q")
            t_q_ht1 = self.target_critic_net.evaluate_q(
                ns_n_t[:, 1:, :], a_n_t[:, 1:, :], t_a_ht1
            )  # t_critic(h_{t+1}, t_actor(h_{t+1})): [h_1(s_1, a_0), h_2(s_2, a_1), ..., h_n(s_n, a_{n-1}), ...]
            self.logger.info(f"critic evaluate_q done, t_q_ht1.shape: {t_q_ht1.shape}")

            # logger.info(f"t_q_ht_bl.shape: {t_q_ht_bl.shape}")
            # y_n_t shape (batch_size, seq_len, 1), target value
            y_n_t = (
                r_n_t[:, 1:, :] + self.hyper_param.Gamma * t_q_ht1
            )  # y0(r_0, Q(h_1,mu(h_1))), y1(r_1, Q(h_2,mu(h_2)), ...)
            self.logger.info(f"y_n_t.shape: {y_n_t.shape}")

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
        self.logger.info(f"applied critic gradient", extra=dictLogger)

        # train actor using bptt
        with tf.GradientTape() as tape:
            self.logger.info(f"start actor evaluate_actions", extra=dictLogger)
            a_ht = self.actor_net.evaluate_actions(
                s_n_t[:, 1:, :], a_n_t[:, :-1, :]
            )  # states, last actions: (s_0, a_-1), (s_1, a_0), ..., (s_n, a_{n-1})
            self.logger.info(
                f"actor evaluate_actions done, a_ht.shape: {a_ht.shape}",
                extra=dictLogger,
            )
            q_ht = self.critic_net.evaluate_q(s_n_t[:, 1:, :], a_n_t[:, :-1, :], a_ht)
            self.logger.info(
                f"actor evaluate_q done, q_ht.shape: {q_ht.shape}",
                extra=dictLogger,
            )
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
        self.logger.info(f"applied actor gradient", extra=dictLogger)

        return actor_loss, critic_loss

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
