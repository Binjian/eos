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
"""
## Introduction

**Recurrent Deterministic Policy Gradient (RDPG)** is a model-free off-policy algorithm for
learning continous actions.

It combines ideas from DDPG (Deep Deterministic Policy Gradient), DQN and RNN.
It uses Experience Replay and slow-learning target networks from DQN, and it is based on
DPG,
which can operate over continuous action spaces.

This tutorial closely follow this paper -
[Continuous control with deep reinforcement learning](https://arxiv.org/pdf/1509.02971.pdf)

## Problem

We are trying to solve the classic **Inverted Pendulum** control problem.
In this setting, we can take only two actions: swing left or swing right.

What make this problem challenging for Q-Learning Algorithms is that actions
are **continuous** instead of being **discrete**. That is, instead of using two
discrete actions like `-1` or `+1`, we have to select from infinite actions
ranging from `-2` to `+2`.

## Quick theory

Just like the Actor-Critic method, we have two networks:

1. Actor - It proposes an action given a state.
2. Critic - It predicts if the action is good (positive value) or bad (negative value)
given a state and an action.

DDPG uses two more techniques not present in the original DQN:

**First, it uses two Target networks.**

**Why?** Because it add stability to training. In short, we are learning from estimated
targets and Target networks are updated slowly, hence keeping our estimated targets
stable.

Conceptually, this is like saying, "I have an idea of how to play this well,
I'm going to try it out for a bit until I find something better",
as opposed to saying "I'm going to re-learn how to play this entire game after every
move".
See this [StackOverflow answer](https://stackoverflow.com/a/54238556/13475679).

**Second, it uses Experience Replay.**

We store list of tuples `(state, action, reward, next_state)`, and instead of
learning only from recent experience, we learn from sampling all of our experience
accumulated so far.

Now, let's see how is it implemented.
"""

"""
We use [OpenAIGym](http://gym.openai.com/docs) to create the environment.
We will use the `upper_bound` parameter to scale our actions later.
"""

"""
The `Buffer` class implements Experience Replay.

---
![Algorithm](https://i.imgur.com/mS6iGyJ.jpg)
---


**Critic loss** - Mean Squared Error of `y - Q(s, a)`
where `y` is the expected return as seen by the Target network,
and `Q(s, a)` is action value predicted by the Critic network. `y` is a moving target
that the critic model tries to achieve; we make this target
stable by updating the Target model slowly.

**Actor loss** - This is computed using the mean of the value given by the Critic network
for the actions taken by the Actor network. We seek to maximize this quantity.

Hence we update the Actor network so that it produces actions that get
the maximum predicted value as seen by the Critic, for a given state.
"""

# system imports
import os

# third-party imports
import numpy as np
import tensorflow as tf
from keras.utils import pad_sequences
from keras import layers
import keras.initializers as initializers

# local imports
from eos import logger, dictLogger
from .actor import ActorNet
from .critic import CriticNet
from eos.utils.exception import ReadOnlyError

global _n_obs, _batch_size

class RDPG:
    def __init__(
        self,
        num_observations,
        obs_len,
        seq_len,
        num_actions,
        buffer_capacity=10000,
        batch_size=4,
        hidden_unitsAC=(256, 256),
        n_layersAC=(2, 2),
        padding_value=0,
        gamma=0.99,
        tauAC=(0.001, 0.001),
        lrAC=(0.001, 0.002),
        datafolder="./",
        ckpt_interval="5",
    ):
        """Initialize the RDPG agent.

        Args:
            num_observations (int): Dimension of the state space.
            padding_value (float): Value to pad the state with, impossible value for observation, action or re
        """
        global _n_obs, _batch_size
        self._num_observations = num_observations
        self._obs_len = obs_len
        self._n_obs = num_observations * obs_len  # 3 * 30
        _n_obs = self._n_obs
        self._n_act = num_actions  # reduced action 5 * 17
        self._seq_len = seq_len
        self._batch_size = batch_size
        _batch_size = self._batch_size
        self._padding_value = padding_value
        self._gamma = tf.cast(gamma, dtype=tf.float32)
        # new data
        self.R = (
            []
        )  # list for dynamic buffer, when saving memory needs to be converted to numpy array
        # Number of "experiences" to store at max
        self._buffer_capacity = buffer_capacity
        self._datafolder = datafolder
        self._ckpt_interval = ckpt_interval
        # Num of tuples to train on.

        self.h_t = None

        # Actor Network (w/ Target Network)
        self.init_ckpt()
        self.actor_net = ActorNet(
            self._n_obs,
            self._n_act,
            hidden_unitsAC[0],
            n_layersAC[0],
            padding_value,
            tauAC[0],
            lrAC[0],
            self._ckpt_actor_dir,
            self._ckpt_interval,
        )

        self.target_actor_net = ActorNet(
            self._n_obs,
            self._n_act,
            hidden_unitsAC[0],
            n_layersAC[0],
            padding_value,
            tauAC[0],
            lrAC[0],
            self._ckpt_actor_dir,
            self._ckpt_interval,
        )
        # clone necessary for the first time training
        self.target_actor_net.clone_weights(self.actor_net)

        # Critic Network (w/ Target Network)

        self.critic_net = CriticNet(
            self._n_obs,
            self._n_act,
            hidden_unitsAC[1],
            n_layersAC[1],
            padding_value,
            tauAC[1],
            lrAC[1],
            self._ckpt_critic_dir,
            self._ckpt_interval,
        )

        self.target_critic_net = CriticNet(
            self._n_obs,
            self._n_act,
            hidden_unitsAC[1],
            n_layersAC[1],
            padding_value,
            tauAC[1],
            lrAC[1],
            self._ckpt_critic_dir,
            self._ckpt_interval,
        )
        # clone necessary for the first time training
        self.target_critic_net.clone_weights(self.critic_net)

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.file_replay = datafolder + "/replay_buffer.npy"
        # Its tells us num of times record() was called.
        self.load_replay_buffer()

    def init_ckpt(self):
        # Actor create or restore from checkpoint
        # add checkpoints manager
        self._ckpt_actor_dir = self._datafolder + "/vb_checkpoints/rdpg_actor"
        try:
            os.makedirs(self._ckpt_actor_dir)
            logger.info(
                "Created checkpoint directory for actor: %s",
                self._ckpt_actor_dir,
                extra=dictLogger,
            )
        except FileExistsError:
            logger.info(
                "Actor checkpoint directory already exists: %s",
                self._ckpt_actor_dir,
                extra=dictLogger,
            )

        # critic create or restore from checkpoint
        # add checkpoints manager
        self._ckpt_critic_dir = self._datafolder + "/vb_checkpoints/rdpg_critic"
        try:
            os.makedirs(self._ckpt_critic_dir)
            logger.info(
                f"Created checkpoint directory for critic: %s",
                self._ckpt_critic_dir,
                extra=dictLogger,
            )
        except FileExistsError:
            logger.info(
                f"Critic checkpoint directory already exists: %s",
                self._ckpt_critic_dir,
                extra=dictLogger,
            )

    def actor_predict(self, obs, t):
        """
        Evaluate the actors given a single observations.
        Batchsize is 1.
        """
        # TODO add sequence padding for variable length sequences?
        if t == 0:
            # initialize with padding values
            # TODO: replace the static self._seq_len with the actual length of the sequence
            # TODO: monitor the length of the sequence so far
            self.obs_t = [obs]
        else:
            self.obs_t.append(obs)

        # self.obs_t = np.ones((1, t + 1, self._n_obs))
        # self.obs_t[0, 0, :] = obs
        # expand the batch dimension and turn obs_t into a numpy array
        input_array = tf.convert_to_tensor(np.expand_dims(np.array(self.obs_t), axis=0), dtype=tf.float32)
        logger.info(f"input_array.shape: {input_array.shape}", extra=dictLogger)
        # action = self.actor_net.predict(input_arra)
        action = self.actor_predict_step(input_array)
        logger.info(f"action.shape: {action.shape}", extra=dictLogger)
        return action

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, 90], dtype=tf.float32)])
    def actor_predict_step(self, obs):
        """
        Evaluate the actors given a single observations.
        Batchsize is 1.
        """
        # logger.info(f"Tracing", extra=dictLogger)
        print("Tracing!")
        action = self.actor_net.predict(obs)
        return action

    def reset_noise(self):
        """reset noise of the moving actor network"""
        self.actor_net.reset_noise()

    def add_to_replay(self, h_t):
        """Add the current h_t to the replay buffer.

        Args:
            h_t (np.array): The current h_t, could be variable length
        """
        # logger.info(
        #     f"h_t list shape: {len(h_t)}X{h_t[-1].shape}.",
        #     extra=dictLogger
        # )
        self.h_t = np.array(h_t)
        logger.info(f"h_t np array shape: {self.h_t.shape}.", extra=dictLogger)
        self.R.append(self.h_t)
        if len(self.R) > self._buffer_capacity:
            self.R.pop(0)
        logger.info(f"Memory length: {len(self.R)}", extra=dictLogger)

    def sample_mini_batch(self):
        """Sample a mini batch from the replay buffer. Add post padding for masking
        Args attributes:
            self.R (list): The replay buffer,
            contains lists of variable lengths of (o_t, a_t, r_t) tuples.

        Returns attributes:
            self.o_n_t: A batch of full padded observation sequence (np.array)
            self.a_n_t: A batch of full padded action sequence (np.array)
            self.r_n_t: A batch of full padded reward sequence(np.array)
            next state observation is
        """
        # Sample random indexes
        record_range = min(len(self.R), self._buffer_capacity)
        logger.info(f"record_range: {record_range}", extra=dictLogger)
        indexes = np.random.choice(record_range, self._batch_size)
        logger.info(f"indexes: {indexes}", extra=dictLogger)
        # logger.info(f"R indices type: {type(indexes)}:{indexes}")
        # mini-batch for Reward, Observation and Action, with keras padding
        # padding automatically expands every sequence to the maximal length by pad_sequences

        # logger.info(f"self.R[0][:,-1]: {self.R[0][:,-1]}", extra=dictLogger)
        r_n_t = [self.R[i][:, -1] for i in indexes]
        # logger.info(f"r_n_t.shape: {len(r_n_t)}X{len(r_n_t[-1])}")
        self.r_n_t = pad_sequences(
            r_n_t,
            padding="post",
            dtype="float32",
            value=self._padding_value,  # impossible value for wh value; 0 would be a possible value
        )  # return numpy array of shape ( batch_size, max(len(r_n_t)))
        # for alignment with critic output with extra feature dimension
        self.r_n_t = tf.convert_to_tensor(np.expand_dims(self.r_n_t, axis=2),dtype=tf.float32)
        # logger.info(f"r_n_t.shape: {self.r_n_t.shape}")

        o_n_l0 = [
            self.R[i][:, 0 : self._n_obs] for i in indexes
        ]  # list of np.array with variable observation length
        o_n_l1 = [
            o_n_l0[i].tolist() for i in np.arange(self._batch_size)
        ]  # list (batch_size) of list (n_obs) of np.array with variable observation length

        try:
            self.o_n_t = np.array(
                [
                    pad_sequences(
                        o_n_l1i,
                        padding="post",
                        dtype="float32",
                        value=self._padding_value,
                    )  # return numpy array
                    for o_n_l1i in o_n_l1
                ]  # return numpy array list
            )
            self.o_n_t = tf.convert_to_tensor(self.o_n_t, dtype=tf.float32)
        except:
            logger.error("Ragged observation state o_n_l1!", extra=dictLogger)
        # logger.info(f"o_n_t.shape: {self.o_n_t.shape}")

        a_n_l0 = [
            self.R[i][:, self._n_obs : self._n_obs + self._n_act] for i in indexes
        ]  # list of np.array with variable action length
        a_n_l1 = [
            a_n_l0[i].tolist() for i in np.arange(self._batch_size)
        ]  # list (batch_size) of list (n_act) of np.array with variable action length

        try:
            self.a_n_t = np.array(
                [
                    pad_sequences(
                        a_n_l1i,
                        padding="post",
                        dtype="float32",
                        value=self._padding_value,
                    )  # return numpy array
                    for a_n_l1i in a_n_l1
                ]  # return numpy array list
            )
            self.a_n_t = tf.convert_to_tensor(self.a_n_t, dtype=tf.float32)
        except:
            logger.error(f"Ragged action state a_n_l1!", extra=dictLogger)
        # logger.info(f"a_n_t.shape: {self.a_n_t.shape}")

    def train(self):
        """
        Train the actor and critic moving network.

        return:
            tuple: (actor_loss, critic_loss)
        """

        self.sample_mini_batch()
        actor_loss, critic_loss = self.train_step(self.r_n_t, self.o_n_t, self.a_n_t)
        return actor_loss, critic_loss

    # @tf.function(input_signature=[tf.TensorSpec(shape=[None,None,1], dtype=tf.float32),
    #                               tf.TensorSpec(shape=[None,None,90], dtype=tf.float32),
    #                               tf.TensorSpec(shape=[None,None,85], dtype=tf.float32)])
    def train_step(self, r_n_t, o_n_t, a_n_t):
        # train critic USING BPTT
        print("Tracing train_step!")
        logger.info(f"start train_step with tracing")
        # logger.info(f"start train_step")
        with tf.GradientTape() as tape:
            # actions at h_t+1
            logger.info(f"start evaluate_actions")
            t_a_ht1 = self.target_actor_net.evaluate_actions(o_n_t)

            # state action value at h_t+1
            # logger.info(f"o_n_t.shape: {self.o_n_t.shape}")
            # logger.info(f"t_a_ht1.shape: {self.t_a_ht1.shape}")
            logger.info(f"start critic evaluate_q")
            t_q_ht1 = self.target_critic_net.evaluate_q(o_n_t, t_a_ht1)
            logger.info(f"critic evaluate_q done, t_q_ht1.shape: {t_q_ht1.shape}")

            # compute the target action value at h_t for the current batch
            # using fancy indexing
            # t_q_ht bootloading value for estimating target action value y_n_t for time h_t+1
            t_q_ht_bl = tf.cast(
                tf.experimental.numpy.append(
                    t_q_ht1[:, 1:, :], np.zeros((self._batch_size, 1, 1)), axis=1,
                ),  # TODO: replace self._seq_len with maximal seq length
                dtype= tf.float32
            )
            # logger.info(f"t_q_ht_bl.shape: {t_q_ht_bl.shape}")
            # y_n_t shape (batch_size, seq_len, 1)
            y_n_t = r_n_t + self._gamma * t_q_ht_bl
            logger.info(f"y_n_t.shape: {y_n_t.shape}")

            # scalar value, average over the batch, time steps
            critic_loss = tf.math.reduce_mean(
                y_n_t - self.critic_net.evaluate_q(o_n_t, a_n_t)
            )
        critic_grad = tape.gradient(
            critic_loss, self.critic_net.eager_model.trainable_variables
        )
        self.critic_net.optimizer.apply_gradients(
            zip(critic_grad, self.critic_net.eager_model.trainable_variables)
        )
        logger.info(f"applied critic gradient", extra=dictLogger)

        # train actor USING BPTT
        with tf.GradientTape() as tape:
            logger.info(f"start actor evaluate_actions", extra=dictLogger)
            a_ht = self.actor_net.evaluate_actions(o_n_t)
            logger.info(f"actor evaluate_actions done, a_ht.shape: {a_ht.shape}", extra=dictLogger)
            q_ht = self.critic_net.evaluate_q(o_n_t, a_ht)
            logger.info(f"actor evaluate_q done, q_ht.shape: {q_ht.shape}", extra=dictLogger)
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
        # TODO check if this is correct. Compare above actor_grad with below
        actor_grad = tape.gradient(
            actor_loss, self.actor_net.eager_model.trainable_variables
        )
        # logger.info(f"action_gradients: {action_gradients}")
        # logger.info(f"actor_grad_weight: {actor_grad_weight} vs actor_grad: {actor_grad}")
        # logger.info(f"The grad diff: {actor_grad - actor_grad_weight}")
        self.actor_net.optimizer.apply_gradients(
            zip(actor_grad, self.actor_net.eager_model.trainable_variables)
        )
        logger.info(f"applied actor gradient", extra=dictLogger)

        return actor_loss, critic_loss

    def notrain(self):
        """
        Purely evaluate the actor and critic networks to  return the losses without Training.

        return:
            tuple: (actor_loss, critic_loss)
        """

        # get critic loss
        # actions at h_t+1
        self.t_a_ht1 = self.target_actor_net.evaluate_actions(self.o_n_t)

        # state action value at h_t+1
        self.t_q_ht1 = self.target_critic_net.evaluate_q(self.o_n_t, self.t_a_ht1)

        # compute the target action value at h_t for the current batch
        # using fancy indexing
        # t_q_ht bootloading value for estimating target action value y_n_t for time h_t+1
        t_q_ht_bl = tf.experimental.numpy.append(
            self.t_q_ht1[:, [1, self._seq_len], :], 0, axis=1
        )
        # y_n_t shape (batch_size, seq_len, 1)
        self.y_n_t = self.r_n_t + self._gamma * t_q_ht_bl

        # scalar value, average over the batch, time steps
        critic_loss = tf.math.reduce_mean(
            self.y_n_t - self.critic_net.evaluate_q(self.o_n_t, self.a_n_t)
        )

        # get  actor loss
        self.a_ht = self.actor_net.evaluate_actions(self.o_n_t)
        self.q_ht = self.critic_net.evaluate_q(self.o_n_t, self.a_ht)

        # -1 because we want to maximize the q_ht
        # scalar value, average over the batch and time steps
        actor_loss = tf.math.reduce_mean(-self.q_ht)

        return actor_loss, critic_loss

    def soft_update_target(self):
        """
        update target networks with tiny tau value, typical value 0.001.
        done after each batch, slowly update target by Polyak averaging.
        """
        self.target_critic_net.soft_update(self.critic_net)
        self.target_actor_net.soft_update(self.actor_net)

    def save_ckpt(self):
        self.actor_net.save_ckpt()
        self.critic_net.save_ckpt()

    def save_replay_buffer(self):
        replay_buffer_npy = np.array(self.R)
        np.save(self.file_replay, replay_buffer_npy)
        logger.info(
            f"saved replay buffer with size : {len(self.R)}",
            extra=dictLogger,
        )

    def load_replay_buffer(self):
        try:
            replay_buffer_npy = np.load(self.file_replay)
            # self.R = replay_buffer_npy.tolist()
            # reload into memory as a list of np arrays for sampling
            self.R = [
                replay_buffer_npy[i, :, :]
                for i in np.arange(replay_buffer_npy.shape[0])
            ]
            logger.info(
                f"loaded last buffer with size: {len(self.R)}, element[0] size: {self.R[0].shape}.",
                extra=dictLogger,
            )
        except IOError:
            logger.info("blank experience", extra=dictLogger)

    @property
    def num_observations(self):
        return self._num_observations

    @num_observations.setter
    def num_observations(self, value):
        raise ReadOnlyError("num_observations is read-only")

    @property
    def obs_len(self):
        return self._obs_len

    @obs_len.setter
    def obs_len(self, value):
        raise ReadOnlyError("obs_len is read-only")

    @property
    def n_obs(self):
        return self._n_obs

    @n_obs.setter
    def n_obs(self, value):
        raise ReadOnlyError("n_obs is read-only")

    @property
    def n_act(self):
        return self._n_act

    @n_act.setter
    def n_act(self, value):
        raise ReadOnlyError("n_act is read-only")

    @property
    def seq_len(self):
        return self._seq_len

    @seq_len.setter
    def seq_len(self, value):
        raise ReadOnlyError("seq_len is read-only")

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        raise ReadOnlyError("batch_size is read-only")

    @property
    def padding_value(self):
        return self._padding_value

    @padding_value.setter
    def padding_value(self, value):
        raise ReadOnlyError("padding_value is read-only")

    @property
    def buffer_capacity(self):
        return self._buffer_capacity

    @buffer_capacity.setter
    def buffer_capcity(self, value):
        raise ReadOnlyError("buffer_capacity is read-only")

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        raise ReadOnlyError("gamma is read-only")
