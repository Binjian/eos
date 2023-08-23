from __future__ import annotations

import logging
import os
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from eos.agent.dpg import DPG
from eos.agent.utils import HyperParamDDPG, OUActionNoise
from eos.data_io.buffer import DaskBuffer, MongoBuffer
from eos.data_io.struct import (
    PoolQuery,  # type: ignore
    veos_lifetime_end_date,
    veos_lifetime_start_date,
)

# from pymongoarrow.monkey import patch_all
from eos.utils import dictLogger, logger

# from tensorflow.python.keras import CheckpointManager, Checkpoint


# patch_all()
"""
Title: DDPG for VEOS
Author: Binjian Xin
Date created: 2021/06/29
Last modified: 2021/06/29
Description: Adapted from keras.io as following


Title: Deep Deterministic Policy Gradient (DDPG)
Author: [amifunny](https://github.com/amifunny)
Date created: 2020/06/04
Last modified: 2020/09/21
Description: Implementing DDPG algorithm on the Inverted Pendulum Problem.

"""
"""
## Introduction

**Deep Deterministic Policy Gradient (DDPG)** is a model-free off-policy algorithm for
learning continuous actions.

It combines ideas from DPG (Deterministic Policy Gradient) and DQN (Deep Q-Network).
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
We use [OpenAIGym](https://gym.openai.com/docs) to create the environment.
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
hyper_param_default = HyperParamDDPG()
actor_optimizer_default = tf.keras.optimizers.Adam(hyper_param_default.ActorLR)  # 0.001
ckpt_actor_default = tf.train.Checkpoint(
    step=tf.Variable(1),
    optimizer=actor_optimizer_default,
    net=tf.keras.Model(),
)
manager_actor_default = tf.train.CheckpointManager(
    ckpt_actor_default, "./actor", max_to_keep=10
)

critic_optimizer_default = tf.keras.optimizers.Adam(
    hyper_param_default.CriticLR
)  # 0.002
ckpt_critic_default = tf.train.Checkpoint(
    step=tf.Variable(1),
    optimizer=critic_optimizer_default,
    net=tf.keras.Model(),
)
manager_critic_default = tf.train.CheckpointManager(
    ckpt_critic_default, "./critic", max_to_keep=10
)


@dataclass
class DDPG(DPG):
    """
    DDPG agent for VEOS.
        data interface:
            - pool in mongodb
            - buffer in memory (numpy array)
        model interface:
            - actor network
            - critic network
    """

    _buffer: Union[
        MongoBuffer, DaskBuffer
    ] = (
        MongoBuffer()
    )  # cannot have default value, because it precedes _plot in base class DPG
    logger: logging.Logger = logging.Logger("eos.agent.ddpg.ddpg")
    _episode_start_dt: datetime = field(default_factory=datetime.now)
    _actor_model: tf.keras.Model = field(default_factory=tf.keras.Model)
    _critic_model: tf.keras.Model = field(default_factory=tf.keras.Model)
    _target_actor_model: tf.keras.Model = field(default_factory=tf.keras.Model)
    _target_critic_model: tf.keras.Model = field(default_factory=tf.keras.Model)
    manager_critic: tf.train.CheckpointManager = manager_critic_default
    ckpt_critic: tf.train.Checkpoint = ckpt_critic_default
    manager_actor: tf.train.CheckpointManager = manager_actor_default
    ckpt_actor: tf.train.Checkpoint = ckpt_actor_default
    actor_saved_model_path: Path = Path("./actor")
    critic_saved_model_path: Path = Path("./critic")

    def __post_init__(self):
        self.logger = logger.getChild("eos").getChild(self.__str__())
        self.logger.propagate = True
        self.dictLogger = dictLogger

        super().__post_init__()  # call DPG post_init for pool init and plot init
        self.coll_type = "RECORD"
        self.hyper_param = HyperParamDDPG(
            CriticStateInputDenseDimension1=16,
            CriticStateInputDenseDimension2=32,
            CriticActionInputDenseDimension=32,
            CriticOutputDenseDimension1=256,
            CriticOutputDenseDimension2=256,
            ActorInputDenseDimension1=256,
            ActorInputDenseDimension2=256,
            BatchSize=4,
            NStates=self.truck.observation_numel,  # 600
            NActions=self.truck.torque_flash_numel,  # 68
            ActionBias=self.truck.torque_bias,  # 0.0
            NLayerActor=2,
            NLayersCritic=2,
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
            timestamp_start=veos_lifetime_start_date,
            timestamp_end=veos_lifetime_end_date,
        )

        # print(f"In DDPG buffer is {self.buffer}!")
        # Initialize networks
        self.actor_model = self.get_actor(
            self.truck.observation_numel,
            self.truck.torque_flash_numel,
            self.hyper_param.ActorInputDenseDimension1,  # 256
            self.hyper_param.ActorInputDenseDimension2,  # 256
            self.hyper_param.NLayerActor,  # 2
            self.hyper_param.ActionBias,  # 0.0
        )

        # Initialize networks
        self.target_actor_model = self.get_actor(
            self.truck.observation_numel,
            self.truck.torque_flash_numel,
            self.hyper_param.ActorInputDenseDimension1,  # 256
            self.hyper_param.ActorInputDenseDimension2,  # 256
            self.hyper_param.NLayerActor,  # 2
            self.hyper_param.ActionBias,  # 0.0
        )

        self.critic_model = self.get_critic(
            self.truck.observation_numel,
            self.truck.torque_flash_numel,
            self.hyper_param.CriticStateInputDenseDimension1,  # 16
            self.hyper_param.CriticStateInputDenseDimension2,  # 32
            self.hyper_param.CriticActionInputDenseDimension,  # 32
            self.hyper_param.CriticOutputDenseDimension1,  # 256
            self.hyper_param.CriticOutputDenseDimension2,  # 256
            self.hyper_param.NLayerCritic,  # 2
        )

        self.target_critic_model = self.get_critic(
            self.truck.observation_numel,
            self.truck.torque_flash_numel,
            self.hyper_param.CriticStateInputDenseDimension1,  # 16
            self.hyper_param.CriticStateInputDenseDimension2,  # 32
            self.hyper_param.CriticActionInputDenseDimension,  # 32
            self.hyper_param.CriticOutputDenseDimension1,  # 256
            self.hyper_param.CriticOutputDenseDimension2,  # 256
            self.hyper_param.NLayerCritic,  # 2
        )

        self.actor_optimizer = tf.keras.optimizers.Adam(
            self.hyper_param.ActorLR
        )  # 0.001
        self.critic_optimizer = tf.keras.optimizers.Adam(
            self.hyper_param.CriticLR
        )  # 0.002

        # ou_noise is a row vector of num_actions dimension
        self.ou_noise_std_dev = 0.2
        self.ou_noise = OUActionNoise(
            mean=np.zeros(self.truck.torque_flash_numel),
            std_deviation=float(self.ou_noise_std_dev)
            * np.ones(self.truck.torque_flash_numel),
        )
        self.init_checkpoint()
        # super().__post_init__()
        self.touch_gpu()

    # def __del__(self):
    #     if self.db_key:
    #         # for database, exit needs drop interface.
    #         self.buffer.drop()
    #     else:
    #         self.buffer.save_replay_buffer()

    def __repr__(self):
        return f"DDPG({self.truck.vid}, {self.driver.pid})"

    def __str__(self):
        return "DDPG"

    def __hash__(self):
        return hash(self.__repr__())

    def init_checkpoint(self):
        # add checkpoints manager
        if self.resume:
            checkpoint_actor_dir = Path(self.data_folder).joinpath(
                "tf_ckpts-"
                + self.__str__()
                + "-"
                + self.truck.vid
                + "-"
                + self.driver.pid
                + "_"
                + "actor"
            )
            checkpoint_critic_dir = Path(self.data_folder).joinpath(
                "tf_ckpts-"
                + self.__str__()
                + "-"
                + self.truck.vid
                + "-"
                + self.driver.pid
                + "_"
                + "critic"
            )
        else:
            checkpoint_actor_dir = Path(self.data_folder).joinpath(
                "tf_ckpts-"
                + self.__str__()
                + "/"
                + self.truck.vid
                + "-"
                + self.driver.pid
                + "_actor"
                + datetime.now().strftime("%y-%m-%d-%H-%M-%S")
            )
            checkpoint_critic_dir = Path(self.data_folder).joinpath(
                "tf_ckpts-"
                + self.__str__()
                + "/"
                + self.truck.vid
                + "-"
                + self.driver.pid
                + "_critic"
                + datetime.now().strftime("%y-%m-%d-%H-%M-%S")
            )
        try:
            os.makedirs(checkpoint_actor_dir)
            self.logger.info(
                f"{{'header': 'Actor folder doesn't exist. Created!'}}",
                extra=dictLogger,
            )
        except FileExistsError:
            self.logger.info(
                f"{{'header': 'Actor folder exists, just resume!'}}",
                extra=dictLogger,
            )
        try:
            os.makedirs(checkpoint_critic_dir)
            self.logger.info(
                f"{{'header': 'Critic folder doesn't exist. Created!'}}",
                extra=dictLogger,
            )
        except FileExistsError:
            self.logger.info(
                f"{{'header': 'Critic folder exists, just resume!'}}",
                extra=dictLogger,
            )

        self.ckpt_actor = tf.train.Checkpoint(
            step=tf.Variable(1, name="step"),
            optimizer=self.actor_optimizer,
            net=self.actor_model,
        )
        self.manager_actor = tf.train.CheckpointManager(
            self.ckpt_actor, checkpoint_actor_dir, max_to_keep=10
        )
        # restore the latest checkpoint to self.actor_model via self.ckpt_actor from self.manager_actor
        self.ckpt_actor.restore(self.manager_actor.latest_checkpoint)
        if self.manager_actor.latest_checkpoint:
            self.logger.info(
                f"{{'header': 'Actor Restored', "
                f"'actor ckpt path': '{self.manager_actor.latest_checkpoint}'}}",
                extra=dictLogger,
            )
        else:
            self.logger.info(
                f"{{'header': 'Actor Initializing from scratch'}}", extra=dictLogger
            )

        self.ckpt_critic = tf.train.Checkpoint(
            step=tf.Variable(1, name="step"),
            optimizer=self.critic_optimizer,
            net=self.critic_model,
        )
        self.manager_critic = tf.train.CheckpointManager(
            self.ckpt_critic, checkpoint_critic_dir, max_to_keep=10
        )
        # restore the latest checkpoint to self.critic_model via self.ckpt_critic from self.manager_critic
        self.ckpt_critic.restore(self.manager_critic.latest_checkpoint)
        if self.manager_critic.latest_checkpoint:
            self.logger.info(
                f"{{'header': 'Critic Restored', "
                f"'critic ckpt path': '{self.manager_critic.latest_checkpoint}'}}",
                extra=dictLogger,
            )
        else:
            self.logger.info(
                "{{'header': 'Critic Initializing from scratch", extra=dictLogger
            )

        # Making the weights equal initially after checkpoints load
        self.target_actor_model.set_weights(self.actor_model.get_weights())
        self.target_critic_model.set_weights(self.critic_model.get_weights())

        self.actor_saved_model_path = Path(self.data_folder).joinpath(
            "tf_ckpts-"
            + self.__str__()
            + "-"
            + self.truck.vid
            + "-"
            + self.driver.pid
            + "_"
            + "actor_saved_model"
        )
        self.critic_saved_model_path = Path(self.data_folder).joinpath(
            "tf_ckpts-"
            + self.__str__()
            + "-"
            + self.truck.vid
            + "-"
            + self.driver.pid
            + "_"
            + "critic_saved_model"
        )

    def save_as_saved_model(self):  # TODO bug fixing
        """save the actor and critic networks as saved model"""

        tf.saved_model.save(self.actor_model, self.actor_saved_model_path)
        tf.saved_model.save(self.critic_model, self.critic_saved_model_path)

    def load_saved_model(self):
        # self.actor_model = tf.saved_model.load(self.actor_saved_model_path)
        # self.target_actor_model.clone_weights(self.actor_model)
        # self.critic_model = tf.saved_model.load(self.actor_saved_model_path)
        # self.target_critic_model.clone_weights(self.critic_model)

        actor_model = tf.saved_model.load(self.actor_saved_model_path)
        critic_model = tf.saved_model.load(self.actor_saved_model_path)
        self.logger.info(f"actor_loaded signatures: {actor_model.signatures.keys()}")
        self.logger.info(f"critic_loaded signatures: {critic_model.signatures.keys()}")

        return actor_model, critic_model

        # convert to tflite

    def convert_to_tflite(self):
        converter = tf.lite.TFLiteConverter.from_saved_model(
            self.actor_saved_model_path
        )
        tflite_model = converter.convert()
        with open("actor_model.tflite", "wb") as f:
            f.write(tflite_model)
        converter = tf.lite.TFLiteConverter.from_saved_model(
            self.critic_saved_model_path
        )
        tflite_model = converter.convert()
        with open("critic_model.tflite", "wb") as f:
            f.write(tflite_model)

        self.model_summary_print(
            self.actor_model, self.actor_saved_model_path / "/../actor_model_spec.txt"
        )
        self.tflite_analytics_print(
            self.actor_saved_model_path / "/../actor_model.tflite"
        )
        self.model_summary_print(
            self.critic_model,
            self.critic_saved_model_path / "/../critic_model_spec.txt",
        )
        self.tflite_analytics_print(
            self.critic_saved_model_path / "/../critic_model.tflite"
        )

    @classmethod
    def model_summary_print(cls, mdl: tf.keras.Model, file_path: Path):
        with file_path.open(mode="w") as f:
            with redirect_stdout(f):
                mdl.summary()

    @classmethod
    def tflite_analytics_print(cls, tflite_file_path: Path):
        with tflite_file_path.open(mode="w") as f:
            with redirect_stdout(f):
                tf.lite.experimental.Analyzer.analyze(tflite_file_path)

    def save_ckpt(self):
        """
        Save checkpoints
        """
        self.ckpt_actor.step.assign_add(1)  # type: ignore
        self.ckpt_critic.step.assign_add(1)  # type: ignore

        if int(self.ckpt_actor.step) % self.hyper_param.CkptInterval == 0:  # type: ignore
            save_path_actor = self.manager_actor.save()
            self.logger.info(
                f"Saved checkpoint for step {int(self.ckpt_actor.step)}: {save_path_actor}",  # type: ignore
                extra=dictLogger,
            )
        if int(self.ckpt_critic.step) % self.hyper_param.CkptInterval == 0:  # type: ignore
            save_path_critic = self.manager_critic.save()
            self.logger.info(
                f"Saved checkpoint for step {int(self.ckpt_actor.step)}: {save_path_critic}",  # type: ignore
                extra=dictLogger,
            )

    @tf.function
    def update_target(self, target_weights, weights, tau):
        for a, b in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    @tf.function
    def soft_update_target(self):
        # This update target parameters slowly
        # Based on rate `tau`, which is much less than one.
        self.update_target(
            self.target_actor_model.variables,
            self.actor_model.variables,
            self.hyper_param.TauActor,
        )
        self.update_target(
            self.target_critic_model.variables,
            self.critic_model.variables,
            self.hyper_param.TauCritic,
        )

    """
    Here we define the Actor and Critic networks. These are basic Dense models
    with `ReLU` activation.
    
    Note: We need the initialization for last layer of the Actor to be between
    `-0.003` and `0.003` as this prevents us from getting `1` or `-1` output values in
    the initial stages, which would squash our gradients to zero,
    as we use the `tanh` activation.
    """

    # action = (Table * Budget + 1); action < action_upper && action > action_lower
    # all in percentage, normalized to initial default table
    # to apply action, first reshape:
    # output actions is a row vector, needs to be reshaped to be a calibration table
    # actions = tf.reshape(get_actors(**), [vcu_calib_table_row, vcu_calib_table_col])\
    # then multiply by default values:
    # actions = tf.math.multiply(actions, vcu_calib_table0)

    @classmethod
    def get_actor(
        cls,
        num_states: int,
        num_actions: int,
        num_actor_inputs1: int = 256,
        num_actor_inputs2: int = 256,
        num_layers: int = 2,
        action_bias: float = 0,
    ):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = keras.layers.Input(shape=(num_states,))
        # dummy rescale to avoid recursive using of inputs, also placeholder for rescaling

        x = keras.layers.Dense(
            num_actor_inputs1,
            activation="relu",
            kernel_initializer=tf.keras.initializers.HeNormal(),
        )(inputs)

        # if n_layers <= 1, the loop will be skipped in default
        for i in range(num_layers - 1):
            x = keras.layers.Dense(
                num_actor_inputs2,
                activation="relu",
                kernel_initializer=tf.keras.initializers.HeNormal(),
            )(x)

        # output layer
        out = keras.layers.Dense(
            num_actions,
            activation="tanh",
            kernel_initializer=last_init,
            bias_initializer=tf.keras.initializers.constant(action_bias),
        )(x)

        # # if our budget is +/-5%, outputs should be [0.95, 1.05]
        # outputs = outputs * action_budget + 1
        #
        # # apply lower and upper bound to the outputs,
        # # typical value is [0.8, 1.0]
        # outputs = tf.clip_by_value(outputs, action_lower, action_upper)
        try:
            eager_model = tf.keras.Model(inputs, out)
        except Exception as e:
            print(f"Exception: {e}")
            print(f"inputs: {inputs}")
            print(f"out: {out}")
            raise e
        # graph_model = tf.function(eager_model)
        return eager_model

    @classmethod
    def get_critic(
        cls,
        num_states: int,
        num_actions: int,
        num_state_input_dense1: int = 16,
        num_state_input_dense2: int = 32,
        num_action_input_dense: int = 32,
        num_output_dense1: int = 256,
        num_output_dense2: int = 256,
        num_layers: int = 2,
    ):
        # State as input
        state_input = keras.layers.Input(shape=(num_states,))
        state_out = keras.layers.Dense(num_state_input_dense1, activation="relu")(
            state_input
        )
        state_out = keras.layers.Dense(num_state_input_dense2, activation="relu")(
            state_out
        )

        # Action as input
        action_input = keras.layers.Input(
            shape=(num_actions,)
        )  # action is defined as flattened.
        action_out = keras.layers.Dense(num_action_input_dense, activation="relu")(
            action_input
        )

        # Both are passed through separate layer before concatenating
        x = keras.layers.Concatenate()([state_out, action_out])

        # if n_layers <= 1, the loop will be skipped in default
        for i in range(num_layers - 1):
            x = keras.layers.Dense(
                num_output_dense1,
                activation="relu",
                kernel_initializer=tf.keras.initializers.HeNormal(),
            )(x)
        x = keras.layers.Dense(
            num_output_dense2,
            activation="relu",
            kernel_initializer=tf.keras.initializers.HeNormal(),
        )(x)

        outputs = keras.layers.Dense(1, activation=None)(x)

        # Outputs single value for give state-action
        eager_model = tf.keras.Model([state_input, action_input], outputs)
        # graph_model = tf.function(eager_model)

        return eager_model

    """
    `policy()` returns an action sampled from our Actor network plus some noise for
    exploration.
    """

    # action outputs and noise object are all row vectors of length 21*17 (r*c), output numpy array
    def policy(self, state: pd.Series):
        # We make sure action is within bounds
        # legal_action = np.clip(sampled_actions, action_lower, action_upper)
        # get flat interleaved (not column-wise stacked) tensor from dataframe
        state_flat = tf.convert_to_tensor(
            state.values
        )  # pd.Series values already flattened.
        states = tf.expand_dims(state_flat, 0)  # motion states is 30*3 matrix
        sampled_actions = self.infer_single_sample(states)
        # return np.squeeze(sampled_actions)  # ? might be unnecessary
        return sampled_actions + self.ou_noise()

    def actor_predict(self, state: pd.Series):
        """
        `actor_predict()` returns an action sampled from our Actor network without noise.
        add optional t just to have uniform interface with rdpg
        """
        return self.policy(state)

    @tf.function
    def infer_single_sample(self, state_flat: tf.Tensor):
        # logger.info(f"Tracing", extra=dictLogger)
        print("Tracing infer!")
        sampled_actions = tf.squeeze(self.actor_model(state_flat))
        # Adding noise to action
        return sampled_actions

    def touch_gpu(self):
        # tf.summary.trace_on(graph=True, profiler=True)
        # ignites manual loading of tensorflow library, to guarantee the real-time processing
        # of first data in main thread
        print("touch gpu in DDPG")
        init_states = pd.Series(
            np.random.rand(self.truck.observation_numel),
        )  # state must have 30*5 (speed, throttle, current, voltage) 5 tuple

        _ = self.policy(init_states)
        self.logger.info(
            f"manual load tf library by calling convert_to_tensor",
            extra=dictLogger,
        )
        self.ou_noise.reset()

        # warm up gpu training graph execution pipeline
        if self.buffer.count() != 0:
            if not self.infer_mode:
                self.logger.info(
                    f"ddpg warm up training!",
                    extra=dictLogger,
                )

                (_, _) = self.train()
                self.update_target(
                    self.target_actor_model.variables,
                    self.actor_model.variables,
                    self.hyper_param.TauActor,  # 0.005
                )
                # self.logger.info(f"Updated target actor", extra=self.dictLogger)
                self.update_target(
                    self.target_critic_model.variables,
                    self.critic_model.variables,
                    self.hyper_param.TauCritic,  # 0.005
                )

                # self.logger.info(f"Updated target critic.", extra=self.dictLogger)
                self.logger.info(
                    f"ddpg warm up training done!",
                    extra=dictLogger,
                )

    def sample_minibatch(self):
        """
        Convert batch type from DataFrames to flattened tensors.
        """
        if (
            self.buffer.count() == 0
        ):  # bootstrap for Episode 0 from the current self.observations list
            self.logger.info(
                f"no data in pool, bootstrap from observation_list, "
                f"truck: {self.truck.vid}, driver: {self.driver.pid}.",
                extra=dictLogger,
            )
            assert (
                len(self.observations) > 0
            ), "no data in temporary buffer self.observations!"

            # sample from self.observations

            batch_idx = np.random.choice(
                len(self.observations), self.hyper_param.BatchSize  # 4
            )
            observation_samples = [
                self.observations[i] for i in batch_idx
            ]  # a sampled list of Series

            idx = pd.IndexSlice
            state = []
            action = []
            reward = []
            nstate = []
            for (
                observation
            ) in (
                observation_samples
            ):  # each observation is a Series, contiguous storage in rows (already flattened!)
                state.append(
                    observation.loc[
                        idx["state", ["velocity", "thrust", "brake"]]
                    ].values
                )
                action.append(
                    observation.loc[idx["action", self.torque_table_row_names]].values
                )
                reward.append(observation.loc[idx["reward", ["work"]]].values)
                nstate.append(
                    observation.loc[
                        idx["nstate", ["velocity", "thrust", "brake"]]
                    ].values
                )

            # convert to tensors by stacking so that the first dimension is batch_size
            states = tf.convert_to_tensor(np.stack(state), dtype=tf.float32)
            actions = tf.convert_to_tensor(np.stack(action), dtype=tf.float32)
            rewards = tf.convert_to_tensor(np.stack(reward), dtype=tf.float32)
            next_states = tf.convert_to_tensor(np.stack(nstate), dtype=tf.float32)

        else:  # otherwise sample from pool, ignoring the current ongoing episode to reduce complexity
            # TODO combine current episode with pool, need evenly sampling pool and list of observations, then combine
            # get sampling range, if not enough data, batch is small
            self.logger.info(
                f"start sample from pool with size: {self.hyper_param.BatchSize}, "
                f"truck: {self.truck.vid}, driver: {self.driver.pid}.",
                extra=dictLogger,
            )

            (
                states,
                actions,
                rewards,
                nstates,
            ) = self.buffer.sample  # for both mongo and arrow pool

            states = tf.convert_to_tensor(states, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.float32)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            next_states = tf.convert_to_tensor(nstates.flatten(), dtype=tf.float32)

        return states, actions, rewards, next_states

    def train(self):
        (
            states,
            actions,
            rewards,
            next_states,
        ) = self.sample_minibatch()

        critic_loss, actor_loss = self.update_with_batch(
            states, actions, rewards, next_states
        )
        return critic_loss, actor_loss

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed-up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update_with_batch(
        self,
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
        training=True,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        print("Tracing update!")
        with tf.GradientTape(watch_accessed_variables=training) as tape:
            target_actions = self.target_actor_model(
                next_state_batch, training=training
            )
            y = (
                reward_batch
                + self.hyper_param.Gamma  # gamma 0.99
                * self.target_critic_model(
                    [next_state_batch, target_actions], training=True
                )
            )
            # ? need to confirm since replay buffer will take max over the actions of Q function.:with
            # future_rewards = self.target_critic(
            #             #     [next_state_batch, target_actions], training=True
            #             # )
            # y = reward_batch + self.gamma * tf.reduce_max(future_rewards, axis = 1)
            # ! the question above is not necessary, since deterministic policy is the maximum!
            critic_value = self.critic_model(
                [state_batch, action_batch], training=training
            )
            # scalar value, average over the batch
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        # logger.info(f"BP done.", extra=dictLogger)
        if training:
            critic_grad = tape.gradient(
                critic_loss, self.critic_model.trainable_variables
            )
            self.critic_optimizer.apply_gradients(
                zip(critic_grad, self.critic_model.trainable_variables)
            )
        else:
            self.logger.info(f"No critic model update!.", extra=dictLogger)

        with tf.GradientTape(watch_accessed_variables=training) as tape:
            actions = self.actor_model(state_batch, training=training)
            critic_value = self.critic_model([state_batch, actions], training=training)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            # scalar value, average over the batch
            actor_loss = -tf.math.reduce_mean(critic_value)

        if training:
            # gradient director directly over actor model weights
            actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
            # TODO Check if this is correct. compare above actor_grad tensor with below
            # action_gradients= tape.gradient(actions, actor_model.trainable_variables)
            # actor_grad = tape.gradient(actor_loss, actions, action_gradients)

            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor_model.trainable_variables)
            )
        else:
            self.logger.info(f"No actor model updates!", extra=dictLogger)

        return critic_loss, actor_loss

    # we only calculate the loss

    # We only compute the loss and don't update parameters
    def get_losses(self):
        (
            states,
            actions,
            rewards,
            next_states,
        ) = self.sample_minibatch()
        critic_loss, actor_loss = self.update_with_batch(
            states, actions, rewards, next_states, training=False
        )
        return critic_loss, actor_loss

    @property
    def actor_model(self) -> tf.keras.Model:
        return self._actor_model

    @actor_model.setter
    def actor_model(self, actor_model: tf.keras.Model):
        self._actor_model = actor_model

    @property
    def critic_model(self) -> tf.keras.Model:
        return self._critic_model

    @critic_model.setter
    def critic_model(self, critic_model: tf.keras.Model):
        self._critic_model = critic_model

    @property
    def target_actor_model(self) -> tf.keras.Model:
        return self._target_actor_model

    @target_actor_model.setter
    def target_actor_model(self, target_actor_model: tf.keras.Model):
        self._target_actor_model = target_actor_model

    @property
    def target_critic_model(self) -> tf.keras.Model:
        return self._target_critic_model

    @target_critic_model.setter
    def target_critic_model(self, target_critic_model: tf.keras.Model):
        self._target_critic_model = target_critic_model
