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


# third-party imports
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
import tensorflow.keras.initializers as initializers

# local imports
from ...l045a_rdpg import logger, logc, logd, dictLogger
from actor import ActorNet
from critic import CriticNet


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
        gammaAC=(0.99, 0.99),
        tauAC=(0.001, 0.001),
        lrAC=(0.001, 0.002),
        datafolder="./",
    ):
        """Initialize the RDPG agent.

        Args:
            num_observations (int): Dimension of the state space.
            padding_value (float): Value to pad the state with, impossible value for observation, action or re
        """

        self.num_observations = num_observations
        self.obs_len = obs_len
        self.n_obs = num_observations * obs_len  # 3 * 30
        self.n_act = num_actions  # reduced action 5 * 17
        self.seq_len = seq_len
        self.data_folder = datafolder
        self.batch_size = batch_size
        self.padding_value = padding_value
        # new data
        self.R = (
            []
        )  # list for dynamic buffer, when saving memory needs to be converted to numpy array
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.

        self.h_t = None

        # old data

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.file_replay = self.data_folder + "/replay_buffer.npy"
        # Its tells us num of times record() was called.
        self.load()

        self.actor_net = ActorNet(
            self.n_obs,
            self.n_act,
            self.seq_len,
            self.batch_size,
            hidden_unitsAC[0],
            n_layersAC[0],
            self.padding_value,
            gammaAC[0],
            tauAC[0],
            lrAC[0],
        )
        self.critic_net = CriticNet(
            self.n_obs,
            self.n_act,
            self.seq_len,
            self.batch_size,
            hidden_unitsAC[1],
            n_layersAC[1],
            self.padding_value,
            gammaAC[1],
            tauAC[1],
            lrAC[1],
        )

        self.target_actor_net = ActorNet(
            self.n_obs,
            self.n_act,
            self.seq_len,
            self.batch_size,
            hidden_unitsAC[0],
            n_layersAC[0],
            self.padding_value,
            gammaAC[0],
            tauAC[0],
            lrAC[0],
        )
        self.target_actor_net.clone_weights(self.actor_net)

        self.target_critic_net = CriticNet(
            self.n_obs,
            self.n_act,
            self.seq_len,
            self.batch_size,
            hidden_unitsAC[1],
            n_layersAC[1],
            self.padding_value,
            gammaAC[1],
            tauAC[1],
            lrAC[1],
        )
        self.target_critic_net.clone_weights(self.critic_net)

    def evaluate_actors(self, obs, t):
        """
        Evaluate the actors given a single observations.
        Batchsize is 1.
        """
        # TODO add sequence padding for variable length sequences?
        if t == 0:
            # initialize with padding values
            self.obs_t = np.ones((1, self.seq_len, self.n_obs)) * self.padding_value
            self.obs_t[0, 0, :] = obs
        else:
            self.obs_t[0, t, :] = obs

        return self.actor_net.predict(self.obs_t)

    def add_to_replay(self, h_t):
        """Add the current h_t to the replay buffer.

        Args:
            h_t (np.array): The current h_t, could be variable length
        """
        self.h_t = h_t
        self.R.append(h_t)
        if len(self.R) > self.buffer_capacity:
            self.R.pop(0)

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
        record_range = min(len(self.R), self.buffer_capacity)
        indexes = np.random.choice(record_range, self.batch_size)

        # mini-batch for Reward, Observation and Action, with keras padding
        self.r_n_t = pad_sequences(
            [self.R[i][:, -1] for i in indexes],
            padding="post",
            dtype="float32",
            value=self.padding_value,  # impossible value for wh value; 0 would be a possible value
        )  # return numpy array of shape (batch_size, seq_len)
        # return numpy array of shape (batch_size, seq_len, 1), for align with critic output with extra feature dimension
        self.r_n_t = np.expand_dims(self.r_n_t, axis=2)

        o_n_l0 = [
            self.R[i][:, 0 : self.n_obs] for i in indexes
        ]  # list of np.array with variable observation length
        o_n_l1 = [
            [o_n_l0[i][:, j] for i in np.arange(self.batch_size)]
            for j in np.arange(self.n_obs)
        ]  # list (batch_size) of list (n_obs) of np.array with variable observation length

        try:
            self.o_n_t = np.array(
                [
                    pad_sequences(
                        o_n_l1i,
                        padding="post",
                        dtype="float32",
                        value=self.padding_value,
                    )  # return numpy array
                    for o_n_l1i in o_n_l1
                ]  # return numpy array list
            )
        except:
            logd.error("Ragged observation state o_n_l1!")

        a_n_l0 = [
            self.R[i][:, self.n_obs : self.n_obs + self.n_act] for i in indexes
        ]  # list of np.array with variable action length
        a_n_l1 = [
            [a_n_l0[i][:, j] for i in np.arange(self.batch_size)]
            for j in np.arange(self.n_act)
        ]  # list (batch_size) of list (n_act) of np.array with variable action length

        try:
            self.a_n_t = np.array(
                [
                    pad_sequences(
                        a_n_l1i,
                        padding="post",
                        dtype="float32",
                        value=self.padding_value,
                    )  # return numpy array
                    for a_n_l1i in a_n_l1
                ]  # return numpy array list
            )
        except:
            logd.error("Ragged action state a_n_l1!")

    def train(self):

        self.sample_mini_batch()

        # train critic USING BPTT
        with tf.GradientTape() as tape:
            # actions at h_t+1
            self.t_a_ht1 = self.target_actor_net.evaluate_actions(self.o_n_t)

            # state action value at h_t+1
            self.t_q_ht1 = self.target_critic_net.evaluate_q(self.o_n_t, self.t_a_ht1)

            # compute the target action value at h_t for the current batch
            # using fancy indexing
            # t_q_ht bootloading value for estimating target action value y_n_t for time h_t+1
            t_q_ht_bl = np.append(self.t_q_ht1[:, [1, self.seq_len], :], 0, axis=1)
            # y_n_t shape (batch_size, seq_len, 1)
            self.y_n_t = self.r_n_t + self.gamma * t_q_ht_bl

            critic_loss = tf.math.reduce_mean(
                self.y_n_t - self.critic_net.evaluate_q(self.o_n_t, self.a_n_t)
            )
        critic_grad = tape.gradient(
            critic_loss, self.critic_net.eager_model.trainable_variables
        )
        self.critic_net.optimizer.apply_gradients(
            zip(critic_grad, self.critic_net.eager_model.trainable_variables)
        )

        # train actor USING BPTT
        with tf.GradientTape() as tape:
            self.a_ht = self.actor_net.evaluate_actions(self.o_n_t)
            self.q_ht = self.critic_net.evaluate_q(self.o_n_t, self.a_ht)

        critic_action_grad = tape.gradient(self.q_ht, self.a_ht) * (
            -1.0
        )  # del_Q_a, -1 to maixmize
        actor_grad = tape.gradient(
            self.a_ht,
            self.actor_net.eager_model.trainable_variables,
            critic_action_grad,  # weights for self.a_ht
        )
        self.actor_net.optimizer.apply_gradients(
            zip(actor_grad, self.actor_net.eager_model.trainable_variables)
        )

        # update target networks with polyak averaging (soft update) need to be done after each batch?
        self.target_critic_net.soft_update(self.critic_net)
        self.target_actor_net.soft_update(self.actor_net)

        return critic_loss

    def save(self):
        replay_buffer_npy = np.array(self.R)
        np.save(self.file_replay, replay_buffer_npy)
        logd.info(
            f"saved replay buffer with size : {len(self.R)}",
            extra=dictLogger,
        )

    def load(self):
        try:
            replay_buffer_npy = np.load(self.file_replay)
            self.R = replay_buffer_npy.tolist()
            logd.info(
                f"loaded last buffer with size: {len(self.R)}",
                extra=dictLogger,
            )
        except IOError:
            logd.info("blank experience", extra=dictLogger)
