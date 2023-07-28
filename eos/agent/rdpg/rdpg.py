# system imports
import os
from dataclasses import dataclass
import logging

# third-party imports
import numpy as np
import pandas as pd
import tensorflow as tf

from keras.utils import pad_sequences  # type: ignore
from pymongoarrow.monkey import patch_all  # type: ignore

# local imports
from eos.utils import dictLogger, logger
from eos.data_io.struct import EpisodeDoc
from ..dpg import DPG  # type: ignore
from ..hyperparams import hyper_param_by_name, HYPER_PARAM

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

    logger: logging.Logger = None
    actor_net: ActorNet = None
    critic_net: CriticNet = None
    target_actor_net: ActorNet = None
    target_critic_net: CriticNet = None
    state_t: list = None
    R: list = None
    h_t: list = None
    buffer_count: int = 0
    _seq_len: int = 8  # length of the sequence for recurrent network
    _ckpt_actor_dir: str = "ckpt_actor"
    _ckpt_critic_dir: str = "ckpt_critic"

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
        self.hyper_param = hyper_param_by_name(self.__class__.__name__)

        # actor network (w/ target network)
        self.init_checkpoint()

        self.actor_net = ActorNet(
            self.truck.observation_numel,
            self.truck.torque_flash_numel,
            self.hyper_param.HiddenUnitsAction,  # 256
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
            self.hyper_param.HiddenUnitsAction,  # 256
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
            self.hyper_param.HiddenUnitsOut,  # 256
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
            self.hyper_param.HiddenUnitsOut,  # 256
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

        _ = self.actor_predict(input_array, 0)
        self.logger.info(
            f"manual load tf library by calling convert_to_tensor",
            extra=self.dictLogger,
        )

        self.actor_net.ou_noise.reset()

        # warm up the gpu training graph execution pipeline
        self.buffer_count = self.buffer.count()
        if self.buffer_count != 0:
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

    def actor_predict(self, state: pd.Series):
        """
        evaluate the actors given a single observations.
        batchsize is 1.
        """

        # self.state_t = np.ones((1, t + 1, self._num_states))
        # self.state_t[0, 0, :] = obs
        # expand the batch dimension and turn obs_t into a numpy array
        input_array = tf.convert_to_tensor(
            np.expand_dims(np.vstack(self.state_t)), dtype=tf.float32
        )
        self.logger.info(
            f"input_array.shape: {input_array.shape}", extra=self.dictLogger
        )
        # action = self.actor_net.predict(input_array)
        action = self.actor_predict_step(input_array)
        self.logger.info(f"action.shape: {action.shape}", extra=self.dictLogger)
        return action

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None, None, 600], dtype=tf.float32)]
    )
    def actor_predict_step(self, obs):
        """
        evaluate the actors given a single observations.
        batchsize is 1.
        """
        # logger.info(f"tracing", extra=self.dictLogger)
        print("tracing!")
        action = self.actor_net.predict(obs)
        return action

    def train(self):
        """
        train the actor and critic moving network.

        return:
            tuple: (actor_loss, critic_loss)
        """

        s_n_t, a_n_t, r_n_t, _ = self.buffer.sample()  # ignore next state for now
        actor_loss, critic_loss = self.train_step(s_n_t, a_n_t, r_n_t)
        return actor_loss, critic_loss

    # @tf.function(input_signature=[tf.tensorspec(shape=[none,none,1], dtype=tf.float32),
    #                               tf.tensorspec(shape=[none,none,90], dtype=tf.float32),
    #                               tf.tensorspec(shape=[none,none,85], dtype=tf.float32)])
    def train_step(self, s_n_t, a_n_t, r_n_t):
        # train critic using bptt
        print("tracing train_step!")
        self.logger.info(f"start train_step with tracing")
        # logger.info(f"start train_step")
        with tf.GradientTape() as tape:
            # actions at h_t+1
            self.logger.info(f"start evaluate_actions")
            t_a_ht1 = self.target_actor_net.evaluate_actions(s_n_t)

            # state action value at h_t+1
            # logger.info(f"o_n_t.shape: {self.o_n_t.shape}")
            # logger.info(f"t_a_ht1.shape: {self.t_a_ht1.shape}")
            self.logger.info(f"start critic evaluate_q")
            t_q_ht1 = self.target_critic_net.evaluate_q(s_n_t, t_a_ht1)
            self.logger.info(f"critic evaluate_q done, t_q_ht1.shape: {t_q_ht1.shape}")

            # compute the target action value at h_t for the current batch
            # using fancy indexing
            # t_q_ht bootloading value for estimating target action value y_n_t for time h_t+1
            t_q_ht_bl = tf.cast(
                tf.experimental.numpy.append(
                    t_q_ht1[:, 1:, :],
                    np.zeros((self.hyper_param.BatchSize, 1, 1)),
                    axis=1,
                ),  # todo: replace self._seq_len with maximal seq length
                dtype=tf.float32,
            )
            # logger.info(f"t_q_ht_bl.shape: {t_q_ht_bl.shape}")
            # y_n_t shape (batch_size, seq_len, 1)
            y_n_t = r_n_t + self.hyper_param.Gamma * t_q_ht_bl
            self.logger.info(f"y_n_t.shape: {y_n_t.shape}")

            # scalar value, average over the batch, time steps
            critic_loss = tf.math.reduce_mean(
                y_n_t - self.critic_net.evaluate_q(s_n_t, a_n_t)
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
            a_ht = self.actor_net.evaluate_actions(s_n_t)
            self.logger.info(
                f"actor evaluate_actions done, a_ht.shape: {a_ht.shape}",
                extra=dictLogger,
            )
            q_ht = self.critic_net.evaluate_q(s_n_t, a_ht)
            self.logger.info(
                f"actor evaluate_q done, q_ht.shape: {q_ht.shape}",
                extra=dictLogger,
            )
            # logger.info(f"a_ht.shape: {self.a_ht.shape}")
            # logger.info(f"q_ht.shape: {self.q_ht.shape}")
            # -1 because we want to maximize the q_ht
            # scalar value, average over the batch and time steps
            actor_loss = tf.math.reduce_mean(-q_ht)

        # action_gradients = tape.gradient(self.a_ht, self.actor_net.eager_model.trainable_variables)
        # actor_grad_weight = tape.gradient(
        #     actor_loss,
        #     self.a_ht,
        #     action_gradients  # weights for self.a_ht
        # )
        # todo check if this is correct. compare above actor_grad with below
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

        s_n_t, a_n_t, r_n_t, _ = self.buffer.sample  # ignore the next state

        # get critic loss
        # actions at h_t+1
        t_a_ht1 = self.target_actor_net.evaluate_actions(s_n_t)

        # state action value at h_t+1
        t_q_ht1 = self.target_critic_net.evaluate_q(s_n_t, t_a_ht1)

        # compute the target action value at h_t for the current batch
        # using fancy indexing
        # t_q_ht bootloading value for estimating target action value y_n_t for time h_t+1
        t_q_ht_bl = tf.experimental.numpy.append(
            t_q_ht1[:, [1, self._seq_len], :], 0, axis=1
        )
        # y_n_t shape (batch_size, seq_len, 1)
        y_n_t = r_n_t + tf.convert_to_tensor(self.hyper_param.Gamma) * t_q_ht_bl

        # scalar value, average over the batch, time steps
        critic_loss = tf.math.reduce_mean(
            y_n_t - self.critic_net.evaluate_q(s_n_t, a_n_t)
        )

        # get  actor loss
        a_ht = self.actor_net.evaluate_actions(s_n_t)
        q_ht = self.critic_net.evaluate_q(s_n_t, a_ht)

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
