# system imports
import os
from datetime import datetime
from dataclasses import dataclass


# third-party imports
import numpy as np
import tensorflow as tf

# from pymongoarrow.api import schema
from keras.utils import pad_sequences
from pymongoarrow.monkey import patch_all


# local imports
from eos import Pool, dictLogger
from eos.config import episode_schemas
from ..dpg import DPG

from .actor import ActorNet
from .critic import CriticNet


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
We DO NOT use [OpenAIGym](https://www.gymlibrary.dev/content/basic_usage/) to create the environment.
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
    _seq_len: int = 8  # length of the sequence for recurrent network
    _ckpt_actor_dir: str = "ckpt_actor"
    _ckpt_critic_dir: str = "ckpt_critic"
    obs_t: list = None
    episode_start_dt: datetime = None
    h_t: list = None
    buffer_counter: int = 0
    r_n_t: tf.Tensor = None
    o_n_t: tf.Tensor = None
    a_n_t: tf.Tensor = None
    R: list = None
    pool: Pool = None

    def __post_init__(
        self,
    ):
        """initialize the rdpg agent.

        args:
            truck.ObservationNumber (int): dimension of the state space.
            padding_value (float): value to pad the state with, impossible value for observation, action or re
        """
        super().__post_init__()

        if self.db:
            self.db_schema = episode_schemas["episode_deep"]
            self.pool = Pool(
                url="mongodb://" + self.db.Host + ":" + self.db.Port,
                username=self.db.Username,
                password=self.db.Password,
                schema=self.db_schema.STRUCTURE,
                db_name=self.db.DatabaseName,
                coll_name=self.db.EpiCollName,
                debug=False,
            )
            self.logger.info(
                f"Connected to MongoDB {self.db.DatabaseName}, collection {self.db.EpiCollName}"
            )
            self.buffer_counter = self.pool.count_items(
                vehicle_id=self.truck.TruckName, driver_id=self.driver
            )
        else:  # elif self.db is '':
            # Instead of list of tuples as the exp.replay concept go
            # We use different np.arrays for each tuple element
            self.file_replay = self.datafolder + "/replay_buffer.npy"
            # Its tells us num of times record() was called.
            self.load_replay_buffer()
            self.buffer_counter = len(self.R)
            self._buffer_capacity = self.buffer_capacity

        # actor network (w/ target network)
        self.init_checkpoint()

        self.actor_net = ActorNet(
            self.num_states,
            self.num_actions,
            self.hidden_unitsAC[0],
            self.n_layersAC[0],
            self.padding_value,
            self.tauAC[0],
            self.lrAC[0],
            self._ckpt_actor_dir,
            self.ckpt_interval,
        )

        self.target_actor_net = ActorNet(
            self.num_states,
            self.num_actions,
            self.hidden_unitsAC[0],
            self.n_layersAC[0],
            self.padding_value,
            self.tauAC[0],
            self.lrAC[0],
            self._ckpt_actor_dir,
            self.ckpt_interval,
        )
        # clone necessary for the first time training
        self.target_actor_net.clone_weights(self.actor_net)

        # critic network (w/ target network)

        self.critic_net = CriticNet(
            self.num_states,
            self.num_actions,
            self.hidden_unitsAC[1],
            self.n_layersAC[1],
            self.padding_value,
            self.tauAC[1],
            self.lrAC[1],
            self._ckpt_critic_dir,
            self.ckpt_interval,
        )

        self.target_critic_net = CriticNet(
            self.num_states,
            self.num_actions,
            self.hidden_unitsAC[1],
            self.n_layersAC[1],
            self.padding_value,
            self.tauAC[1],
            self.lrAC[1],
            self._ckpt_critic_dir,
            self.ckpt_interval,
        )
        # clone necessary for the first time training
        self.target_critic_net.clone_weights(self.critic_net)
        self.touch_gpu()


    def __del__(self):
        if self.db_server:
            # for database, exit needs drop interface.
            self.pool.drop_mongo()
        else:
            self.save_replay_buffer()

    def __repr__(self):

        return f"RDPG({self.truck.name}, {self.driver})"

    def __str__(self):
        return "RDPG"

    def touch_gpu(self):
        # tf.summary.trace_on(graph=true, profiler=true)
        # ignites manual loading of tensorflow library, \
        # to guarantee the real-time processing of first data in main thread
        init_motion_power = np.random.rand(self.num_states)
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
        if self.buffer_counter != 0:
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
            self._datafolder
            + "-"
            + self.__str__()
            + "-"
            + self.truck.TruckName
            + "-"
            + self.driver
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
            self._datafolder
            + "-"
            + self.__str__()
            + "-"
            + self.truck.TruckName
            + "-"
            + self.driver
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

    def actor_predict(self, obs, t):
        """
        evaluate the actors given a single observations.
        batchsize is 1.
        """
        # todo add sequence padding for variable length sequences?
        if t == 0:
            # initialize with padding values
            # todo: replace the static self._seq_len with the actual length of the sequence
            # todo: monitor the length of the sequence so far
            self.obs_t = [obs]
        else:
            self.obs_t.append(obs)

        # self.obs_t = np.ones((1, t + 1, self._num_states))
        # self.obs_t[0, 0, :] = obs
        # expand the batch dimension and turn obs_t into a numpy array
        input_array = tf.convert_to_tensor(
            np.expand_dims(np.array(self.obs_t), axis=0), dtype=tf.float32
        )
        self.logger.info(
            f"input_array.shape: {input_array.shape}", extra=self.dictLogger
        )
        # action = self.actor_net.predict(input_arra)
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

    def start_episode(self, dt: datetime):
        self.logger.info(f"episode start at {dt}", extra=self.dictLogger)
        # somehow mongodb does not like microseconds in rec['plot']
        dt_milliseconds = int(dt.microsecond / 1000) * 1000
        self.episode_start_dt = dt.replace(microsecond=dt_milliseconds)
        self.h_t = []

    def deposit(self, prev_ts, prev_o_t, prev_a_t, prev_table_start, cycle_reward, o_t):
        """deposit the experience into the replay buffer.
        the following are not used for rdpg,
        just to have a uniform interface with ddpg
        prev_ts: timestamp of the previous state
        o_t: current state
        """
        _ = prev_ts
        _ = o_t
        if self.db:
            if not self.h_t:  # first even step has $r_0$
                self.h_t = [
                    {
                        "states": prev_o_t.numpy().tolist(),
                        "actions": prev_a_t.numpy().tolist(),
                        "action_start_row": prev_table_start,
                        "reward": cycle_reward.numpy().tolist(),
                    }
                ]
            else:
                self.h_t.append(
                    {
                        "states": prev_o_t.numpy().tolist(),
                        "actions": prev_a_t.numpy().tolist(),
                        "action_start_row": prev_table_start,
                        "reward": cycle_reward.numpy().tolist(),
                    }
                )

            self.logger.info(
                f"prev_o_t shape: {prev_o_t.shape},prev_a_t shape: {prev_a_t.shape}.",
                extra=self.dictLogger,
            )
        else:  # local buffer needs array
            if not self.h_t:  # first even step has $r_0$
                self.h_t = [np.hstack([prev_o_t, prev_a_t, cycle_reward])]
            else:
                self.h_t.append(np.hstack([prev_o_t, prev_a_t, cycle_reward]))

            self.logger.info(
                f"prev_o_t.shape: {prev_o_t.shape}, prev_a_t.shape: {prev_a_t.shape}, "
                f"cycle_reward: {cycle_reward.shape}, self.h_t shape: {len(self.h_t)}x{self.h_t[-1].shape}.",
                extra=dictLogger,
            )

    def end_episode(self):
        """deposit the experience into the replay buffer."""
        self.deposit_history()
        self.logger.info(f"episode end at {datetime.now()}", extra=self.dictLogger)

    def deposit_history(self):
        """deposit the episode history into the agent replay buffer."""
        if self.db:
            if self.h_t:
                episode = {
                    "timestamp": self.episode_start_dt,
                    "plot": {
                        "character": self.truck.truckname,
                        "driver": self.driver,
                        "when": self.episode_start_dt,
                        "tz": str(self.truck.tz),
                        "where": "campus",
                        "length": len(self.h_t),
                        "states": {
                            "observations": [{"velocity_unit": "kmph"},
                                             {"thrust_unit": "percentage"},
                                             {"brake_unit": "percentage"}],
                            "unit_number": self.truck.CloudUnitNumber,  # 4
                            "unit_duration": self.truck.CloudUnitDuration,  # 1s
                            "frequency": self.truck.CloudSignalFrequency,  # 50 hz
                        },
                        "actions": {
                            "action_row_number": self.truck.actionflashrow,
                            "action_column_number": self.truck.pedalscale,
                        },
                        "reward": {
                            "reward_unit": "wh",
                        },
                    },
                    "history": self.h_t,
                }
                self.add_to_replay_db(episode)
                self.logger.info(
                    f"add episode history to db replay buffer!", extra=self.dictLogger
                )
            else:
                self.logger.info(
                    f"episode done but history is empty or no observation received!",
                    extra=self.dictLogger,
                )

        else:
            self.add_to_replay_buffer(self.h_t)
            self.logger.info(
                f"add episode history to npy replay buffer!", extra=self.dictLogger
            )

    def add_to_replay_db(self, episode):
        """add an episode to database

        db buffer is lists of lists
        """

        self.logger.info("start deposit an episode", extra=self.dictLogger)
        result = self.pool.deposit_item(episode)
        self.logger.info("episode inserted.", extra=self.dictLogger)
        assert result.acknowledged is True, "deposit result not acknowledged"
        self.buffer_counter = self.pool.count_items(self.truck.truckname, self.driver)
        self.logger.info(
            f"pool has {self.buffer_counter} records", extra=self.dictLogger
        )
        epi_inserted = self.pool.find_item(result.inserted_id)
        self.logger.info("episode found.", extra=self.dictLogger)
        assert epi_inserted["timestamp"] == episode["timestamp"], "timestamp mismatch"
        assert epi_inserted["plot"] == episode["plot"], "plot mismatch"
        assert epi_inserted["history"] == episode["history"], "history mismatch"

    def add_to_replay_buffer(self, h_t):
        """add the current h_t to the replay buffer.

        replay buffer is list of 2d numpy arrays
        args:
            h_t (np.array): the current h_t, could be variable length
                            a two-dimensional array of shape (t, 3) with t the number of steps/rows
        """
        # logger.info(
        #     f"h_t list shape: {len(h_t)}x{h_t[-1].shape}.",
        #     extra=self.dictLogger
        # )
        np_h_t = np.array(h_t)
        self.logger.info(
            f"h_t np array shape: {np_h_t.shape}.", extra=self.dictLogger
        )
        self.R.append(np_h_t)
        if len(self.R) > self._buffer_capacity:
            self.R.pop(0)
        self.logger.info(f"memory length: {len(self.R)}", extra=self.dictLogger)

    def sample_mini_batch_from_db(self):
        """sample a mini batch from the database.

        db buffer is lists of lists
        """
        self.logger.info(
            "start sampling a mini batch from the database.", extra=self.dictLogger
        )

        assert self.buffer_counter > 0, "pool is empty!"
        batch = self.pool.sample_batch_items(
            batch_size=self.batch_size,
            vehicle_id=self.truck.truckname,
            driver_id=self.driver,
        )
        assert (
            len(batch) == self.batch_size
        ), f"sampled batch size {len(batch)} not match sample size {self.batch_size}"
        self.logger.info(
            f"{self.batch_size} episodes sampled from {self.buffer_counter}.",
            extra=self.dictLogger,
        )

        # get dimension of the history
        _ = batch[0]["plot"]["length"]
        states_length = batch[0]["plot"]["states"]["length"]
        action_row_number = batch[0]["plot"]["actions"]["action_row_number"]
        action_column_number = batch[0]["plot"]["actions"]["action_column_number"]

        assert (
            self.num_states == states_length * self.truck.ObservationNumber
        ), f"num_states {self.num_states} doesn't match config {states_length} * {self.truck.ObservationNumber}"
        # (3s*50)*3(obs_num))=450
        assert (
            self.num_actions == action_row_number * action_column_number
        ), f"num_actions {self.num_actions} doesn't match config {action_row_number} * {action_column_number}"
        # (3s*50)*3(obs_num))=450

        # decode and padding rewards, states and actions
        # decode reward series
        r_n_t = [
            [history["reward"] for history in episode["history"]] for episode in batch
        ]  # list of lists
        np_r_n_t = pad_sequences(
            r_n_t,
            padding="post",
            dtype="float32",
            value=self._padding_value,
        )

        # for alignment with critic output with extra feature dimension
        self.r_n_t = tf.convert_to_tensor(
            np.expand_dims(np_r_n_t, axis=2), dtype=tf.float32
        )
        self.logger.info(f"r_n_t.shape: {self.r_n_t.shape}")
        # self.logger.info("done decoding reward.", extra=self.dictLogger)

        #  history['states'] for history in episdoe["history"] is the time sequence of states
        o_n_l0 = [
            [history["states"] for history in episode["history"]] for episode in batch
        ]

        # state in o_n_l0 is the time sequence of states [o1, o2, o3, ..., o7]
        # o1=[v0, t0, b0, v1, t1, b1, ...] (3x50x3=450)
        # state has n steps (for example 7)
        # each step has dimension of num_states(450)
        # [step[i] for step in state] is the time sequence of the i-th feature
        o_n_l1 = [
            [[step[i] for step in state] for state in o_n_l0]
            for i in np.arange(self.num_states)
        ]  # list (num_states) of lists (batch_size) of lists with variable observation length

        try:
            o_n_t = np.array(
                [
                    pad_sequences(
                        o_n_l1i,
                        padding="post",
                        dtype="float32",
                        value=self._padding_value,
                    )  # return numpy array
                    for o_n_l1i in o_n_l1
                ]  # return numpy array list
            )  # return numpy array list of size (num_states, batch_size, max(len(o_n_l1i))),
            # max(len(o_n_l1i)) is the longest sequence in the batch, should be the same for all observations
            # otherwise observation is ragged, throw exception
            o_n_t = tf.transpose(
                o_n_t, perm=[1, 2, 0]
            )  # return numpy array list of size (batch_size,max(len(o_n_l1i)), num_states)
            self.o_n_t = tf.convert_to_tensor(o_n_t, dtype=tf.float32)
        except Exception as X:
            self.logger.error(f"ragged observation state o_n_l1; Exception: {X}!", extra=dictLogger)
        self.logger.info(f"o_n_t.shape: {self.o_n_t.shape}")

        # decode starting row series, not used for now
        a_n_start_t = [
            [history["action_start_row"] for history in episode["history"]]
            for episode in batch
        ]
        _ = pad_sequences(
            a_n_start_t,
            padding="post",
            dtype="float32",
            value=self._padding_value,
        )  # a_n_start_t1
        self.logger.info("done decoding starting row.", extra=self.dictLogger)

        # decode action series, not used for now
        a_n_l0 = [[obs["actions"] for obs in episode["history"]] for episode in batch]
        a_n_l1 = [
            [[step[i] for step in act] for act in a_n_l0]
            for i in np.arange(self.num_actions)
        ]  # list (num_actions) of lists (batch_size) of lists with variable observation length

        try:
            a_n_t = np.array(
                [
                    pad_sequences(
                        a_n_l1i,
                        padding="post",
                        dtype="float32",
                        value=self._padding_value,
                    )  # return numpy array
                    for a_n_l1i in a_n_l1
                ]  # return numpy array list
            )  # return numpy array list of size (num_states, batch_size, max(len(o_n_l1i))),
            # max(len(o_n_l1i)) is the longest sequence in the batch, should be the same for all observations
            # otherwise observation is ragged, throw exception
            a_n_t = tf.transpose(
                a_n_t, perm=[1, 2, 0]
            )  # return numpy array list of size (batch_size,max(len(o_n_l1i)), num_states)
            self.a_n_t = tf.convert_to_tensor(a_n_t, dtype=tf.float32)
        except Exception as X:
            self.logger.error(f"ragged action state a_n_l1; Exeception: {X}!", extra=dictLogger)
        self.logger.info(f"a_n_t.shape: {self.a_n_t.shape}")

    def sample_mini_batch_from_buffer(self):
        """sample a mini batch from the replay buffer. add post padding for masking

        replay buffer is list of 2d numpy arrays
        args attributes:
            self.R (list): the replay buffer,
            contains lists of variable lengths of (o_t, a_t, r_t) tuples.

        returns attributes:
            self.o_n_t: a batch of full padded observation sequence (np.array)
            self.a_n_t: a batch of full padded action sequence (np.array)
            self.r_n_t: a batch of full padded reward sequence(np.array)
            next state observation is
        """
        # sample random indexes
        record_range = min(len(self.R), self._buffer_capacity)
        self.logger.info(f"record_range: {record_range}", extra=dictLogger)
        indexes = np.random.choice(record_range, self.batch_size)
        self.logger.info(f"indexes: {indexes}", extra=dictLogger)
        # logger.info(f"r indices type: {type(indexes)}:{indexes}")
        # mini-batch for reward, observation and action
        # with keras padding automatically expands every sequence to the maximal length by pad_sequences

        # logger.info(f"self.R[0][:,-1]: {self.R[0][:,-1]}", extra=dictLogger)
        r_n_t = [self.R[i][:, -1] for i in indexes]  # list of arrays

        # logger.info(f"r_n_t.shape: {len(r_n_t)}x{len(r_n_t[-1])}")
        np_r_n_t = pad_sequences(
            r_n_t,
            padding="post",
            dtype="float32",
            value=self._padding_value,  # impossible value for wh value; 0 would be a possible value
        )  # return numpy array of shape ( batch_size, max(len(r_n_t)))
        # it works for list of arrays! not necessary the following
        # r_n_t = [(self.R[i][:, -1]).tolist() for i in indexes]  # list of arrays

        # for alignment with critic output with extra feature dimension
        self.r_n_t = tf.convert_to_tensor(
            np.expand_dims(np_r_n_t, axis=2), dtype=tf.float32
        )
        # logger.info(f"r_n_t.shape: {self.r_n_t.shape}")

        o_n_l0 = [
            self.R[i][:, 0: self.num_states] for i in indexes
        ]  # list of np.array with variable observation length
        # o_n_l1 = [
        #     o_n_l0[i].tolist() for i in np.arange(self._batch_size)
        # ]  # list (batch_size) of list (num_states) of np.array with variable observation length

        # state[:,i].tolist() is the time sequence of the i-th observation (dimension time)
        # [state[:,i].tolist() for state in o_n_l0] is the list of time sequences of a single batch (dimension batch)
        # o_n_l1 is the final ragged list (different time steps) of different observations (dimension observation)
        o_n_l1 = [
            [state[:, i].tolist() for state in o_n_l0]
            for i in np.arange(self.num_states)
        ]  # list (num_states) of lists (batch_size) of lists with variable observation length

        try:
            o_n_t = np.array(
                [
                    pad_sequences(
                        o_n_l1i,
                        padding="post",
                        dtype="float32",
                        value=self._padding_value,
                    )  # return numpy array
                    for o_n_l1i in o_n_l1
                ]  # return numpy array list
            )  # return numpy array list of size (num_states, batch_size, max(len(o_n_l1i))),
            # max(len(o_n_l1i)) is the longest sequence in the batch, should be the same for all observations
            # otherwise observation is ragged, throw exception
            o_n_t = tf.transpose(
                o_n_t, perm=[1, 2, 0]
            )  # return numpy array list of size (batch_size,max(len(o_n_l1i)), num_states)
            self.o_n_t = tf.convert_to_tensor(o_n_t, dtype=tf.float32)
        except Exception as X:
            self.logger.error(f"ragged observation state o_n_l1; Exception: {X}!", extra=dictLogger)
        # logger.info(f"o_n_t.shape: {self.o_n_t.shape}")

        a_n_l0 = [
            self.R[i][:, self.num_states: self.num_states + self.num_actions]
            for i in indexes
        ]  # list of np.array with variable action length
        # a_n_l1 = [
        #     a_n_l0[i].tolist() for i in np.arange(self._batch_size)
        # ]  # list (batch_size) of list (num_actions) of np.array with variable action length
        a_n_l1 = [
            [act[:, i].tolist() for act in a_n_l0] for i in np.arange(self.num_actions)
        ]  # list (num_actions) of lists (batch_size) of lists with variable observation length

        try:
            a_n_t = np.array(
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
            a_n_t = tf.transpose(
                a_n_t, perm=[1, 2, 0]
            )  # return numpy array list of size (batch_size,max(len(a_n_l1i)), num_actions)
            self.a_n_t = tf.convert_to_tensor(a_n_t, dtype=tf.float32)
        except Exception as X:
            self.logger.error(f"ragged action state a_n_l1; Exception: {X}!", extra=dictLogger)
        # logger.info(f"a_n_t.shape: {self.a_n_t.shape}")

    def train(self):
        """
        train the actor and critic moving network.

        return:
            tuple: (actor_loss, critic_loss)
        """

        if self.db:
            self.sample_mini_batch_from_db()
        else:
            self.sample_mini_batch_from_buffer()
        actor_loss, critic_loss = self.train_step(self.r_n_t, self.o_n_t, self.a_n_t)
        return actor_loss, critic_loss

    # @tf.function(input_signature=[tf.tensorspec(shape=[none,none,1], dtype=tf.float32),
    #                               tf.tensorspec(shape=[none,none,90], dtype=tf.float32),
    #                               tf.tensorspec(shape=[none,none,85], dtype=tf.float32)])
    def train_step(self, r_n_t, o_n_t, a_n_t):
        # train critic using bptt
        print("tracing train_step!")
        self.logger.info(f"start train_step with tracing")
        # logger.info(f"start train_step")
        with tf.GradientTape() as tape:
            # actions at h_t+1
            self.logger.info(f"start evaluate_actions")
            t_a_ht1 = self.target_actor_net.evaluate_actions(o_n_t)

            # state action value at h_t+1
            # logger.info(f"o_n_t.shape: {self.o_n_t.shape}")
            # logger.info(f"t_a_ht1.shape: {self.t_a_ht1.shape}")
            self.logger.info(f"start critic evaluate_q")
            t_q_ht1 = self.target_critic_net.evaluate_q(o_n_t, t_a_ht1)
            self.logger.info(f"critic evaluate_q done, t_q_ht1.shape: {t_q_ht1.shape}")

            # compute the target action value at h_t for the current batch
            # using fancy indexing
            # t_q_ht bootloading value for estimating target action value y_n_t for time h_t+1
            t_q_ht_bl = tf.cast(
                tf.experimental.numpy.append(
                    t_q_ht1[:, 1:, :],
                    np.zeros((self._batch_size, 1, 1)),
                    axis=1,
                ),  # todo: replace self._seq_len with maximal seq length
                dtype=tf.float32,
            )
            # logger.info(f"t_q_ht_bl.shape: {t_q_ht_bl.shape}")
            # y_n_t shape (batch_size, seq_len, 1)
            y_n_t = r_n_t + self._gamma * t_q_ht_bl
            self.logger.info(f"y_n_t.shape: {y_n_t.shape}")

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
        self.logger.info(f"applied critic gradient", extra=dictLogger)

        # train actor using bptt
        with tf.GradientTape() as tape:
            self.logger.info(f"start actor evaluate_actions", extra=dictLogger)
            a_ht = self.actor_net.evaluate_actions(o_n_t)
            self.logger.info(
                f"actor evaluate_actions done, a_ht.shape: {a_ht.shape}",
                extra=dictLogger,
            )
            q_ht = self.critic_net.evaluate_q(o_n_t, a_ht)
            self.logger.info(
                f"actor evaluate_q done, q_ht.shape: {q_ht.shape}", extra=dictLogger
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

        if self.db:
            self.sample_mini_batch_from_db()
        else:
            self.sample_mini_batch()

        # get critic loss
        # actions at h_t+1
        t_a_ht1 = self.target_actor_net.evaluate_actions(self.o_n_t)

        # state action value at h_t+1
        t_q_ht1 = self.target_critic_net.evaluate_q(self.o_n_t, t_a_ht1)

        # compute the target action value at h_t for the current batch
        # using fancy indexing
        # t_q_ht bootloading value for estimating target action value y_n_t for time h_t+1
        t_q_ht_bl = tf.experimental.numpy.append(
            t_q_ht1[:, [1, self._seq_len], :], 0, axis=1
        )
        # y_n_t shape (batch_size, seq_len, 1)
        y_n_t = self.r_n_t + tf.convert_to_tensor(self._gamma) * t_q_ht_bl

        # scalar value, average over the batch, time steps
        critic_loss = tf.math.reduce_mean(
            y_n_t - self.critic_net.evaluate_q(self.o_n_t, self.a_n_t)
        )

        # get  actor loss
        a_ht = self.actor_net.evaluate_actions(self.o_n_t)
        q_ht = self.critic_net.evaluate_q(self.o_n_t, a_ht)

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

    def save_replay_buffer(self):
        replay_buffer_npy = np.array(self.R)
        np.save(self.file_replay, replay_buffer_npy)
        self.logger.info(
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
            self.logger.info(
                f"loaded last buffer with size: {len(self.R)}, element[0] size: {self.R[0].shape}.",
                extra=dictLogger,
            )
        except IOError:
            self.logger.info("blank experience", extra=dictLogger)
            self.R = []
