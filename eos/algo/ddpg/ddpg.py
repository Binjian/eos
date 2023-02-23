from datetime import datetime
from dataclasses import dataclass
import os
import numpy as np
import tensorflow as tf

from keras import layers
from pymongoarrow.monkey import patch_all

from eos import Pool, dictLogger
from eos.config import record_schemas
from ..utils import OUActionNoise
from .buffer import Buffer
from ..dpg import DPG

patch_all()
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
learning continous actions.

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


@dataclass
class DDPG(DPG):
    """
    RDPG agent for VEOS.
        data interface:
            - pool in mongodb
            - buffer in memory (numpy array)
        model interface:
            - actor network
            - critic network
    """
    actor_model: tf.keras.Model = None
    target_actor_model: tf.keras.Model = None
    critic_model: tf.keras.Model = None
    target_critic_model: tf.keras.Model = None
    manager_critic: tf.train.CheckpointManager = None
    ckpt_critic: tf.train.Checkpoint = None
    manager_actor: tf.train.CheckpointManager = None
    ckpt_actor: tf.train.Checkpoint = None

    def __post__init__(self):
        super().__post_init__()

        self.buffer = Buffer(
            self.truck,
            self.driver,
            self.num_states,
            self.num_actions,
            batch_size=self.batch_size,
            datafolder=str(self.datafolder),
            db_server=self.db_server,
        )

        # Initialize networks
        self.actor_model = self.get_actor(
            self.num_states,
            self.num_actions,
            self.hidden_unitsAC[0],
            self.n_layersAC[0],
            self.action_bias,
        )

        # Initialize networks
        self.target_actor = self.get_actor(
            self.num_states,
            self.num_actions,
            self.hidden_unitsAC[0],
            self.n_layersAC[0],
            self.action_bias,
        )

        self.critic_model = self.get_critic(
            self.num_states,
            self.num_actions,
            self.hidden_unitsAC[1],
            self.hidden_unitsAC[2],
            self.hidden_unitsAC[0],
            self.n_layersAC[1],
        )

        self.target_critic = self.get_critic(
            self.num_states,
            self.num_actions,
            self.hidden_unitsAC[1],
            self.hidden_unitsAC[2],
            self.hidden_unitsAC[0],
            self.n_layersAC[1],
        )

        self.actor_optimizer = tf.keras.optimizers.Adam(self.lrAC[0])
        self.critic_optimizer = tf.keras.optimizers.Adam(self.lrAC[1])


        # ou_noise is a row vector of num_actions dimension
        self.ou_noise_std_dev = 0.2
        self.ou_noise = OUActionNoise(
            mean=np.zeros(self.num_actions),
            std_deviation=float(self.ou_noise_std_dev) * np.ones(self.num_actions),
        )
        self.init_checkpoint()
        self.touch_gpu()

    def __repr__(self):
        return f"DDPG({self.truck.name}, {self.driver})"

    def __str__(self):
        return "DDPG"


    def init_checkpoint(self):
        # add checkpoints manager
        if self.resume:
            checkpoint_actor_dir = self.datafolder.joinpath(
                "tf_ckpts-"
                + self.__str__()
                + "-"
                + self.truck.TruckName
                + "-"
                + self.driver
                + "_"
                + "actor"
            )
            checkpoint_critic_dir = self.datafolder.joinpath(
                "tf_ckpts-"
                + self.__str__()
                + "-"
                + self.truck.TruckName
                + "-"
                + self.driver
                + "_"
                + "critic"
            )
        else:
            checkpoint_actor_dir = self.datafolder.joinpath(
                "tf_ckpts-"
                + self.__str__()
                + "/"
                + self.truck.TruckName
                + "-"
                + self.driver
                + "_actor"
                + datetime.now().strftime("%y-%m-%d-%H-%M-%S")
            )
            checkpoint_critic_dir = self.datafolder.joinpath(
                "tf_ckpts-"
                + self.__str__()
                + "/"
                + self.truck.TruckName
                + "-"
                + self.driver
                + "_critic"
                + datetime.now().strftime("%y-%m-%d-%H-%M-%S")
            )
        try:
            os.makedirs(checkpoint_actor_dir)
            self.logger.info("Actor folder doesn't exist. Created!", extra=dictLogger)
        except FileExistsError:
            self.logger.info("Actor folder exists, just resume!", extra=dictLogger)
        try:
            os.makedirs(checkpoint_critic_dir)
            self.logger.info("Critic folder doesn't exist. Created!", extra=dictLogger)
        except FileExistsError:
            self.logger.info("Critic folder exists, just resume!", extra=dictLogger)

        self.ckpt_actor = tf.train.Checkpoint(
            step=tf.Variable(1), optimizer=self.actor_optimizer, net=self.actor_model
        )
        self.manager_actor = tf.train.CheckpointManager(
            self.ckpt_actor, checkpoint_actor_dir, max_to_keep=10
        )
        self.ckpt_actor.restore(self.manager_actor.latest_checkpoint)
        if self.manager_actor.latest_checkpoint:
            self.logger.info(
                f"Actor Restored from {self.manager_actor.latest_checkpoint}",
                extra=dictLogger,
            )
        else:
            self.logger.info(f"Actor Initializing from scratch", extra=dictLogger)

        self.ckpt_critic = tf.train.Checkpoint(
            step=tf.Variable(1), optimizer=self.critic_optimizer, net=self.critic_model
        )
        self.manager_critic = tf.train.CheckpointManager(
            self.ckpt_critic, checkpoint_critic_dir, max_to_keep=10
        )
        self.ckpt_critic.restore(self.manager_critic.latest_checkpoint)
        if self.manager_critic.latest_checkpoint:
            self.logger.info(
                f"Critic Restored from {self.manager_critic.latest_checkpoint}",
                extra=dictLogger,
            )
        else:
            self.logger.info("Critic Initializing from scratch", extra=dictLogger)

        # Making the weights equal initially after checkpoints load
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

    def save_ckpt(self):
        """
        Save checkpoints
        """
        self.ckpt_actor.step.assign_add(1)
        self.ckpt_critic.step.assign_add(1)

        if int(self.ckpt_actor.step) % self.ckpt_interval == 0:
            save_path_actor = self.manager_actor.save()
            self.logger.info(
                f"Saved checkpoint for step {int(self.ckpt_actor.step)}: {save_path_actor}",
                extra=dictLogger,
            )
        if int(self.ckpt_critic.step) % self.ckpt_interval == 0:
            save_path_critic = self.manager_critic.save()
            self.logger.info(
                f"Saved checkpoint for step {int(self.ckpt_actor.step)}: {save_path_critic}",
                extra=dictLogger,
            )

    def save_replay_buffer(self):
        self.buffer.save_replay_buffer()

    def load_replay_buffer(self):
        self.buffer.load_replay_buffer()

    @tf.function
    def update_target(self, target_weights, weights, tau):
        for a, b in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    @tf.function
    def soft_update_target(self):
        # This update target parameters slowly
        # Based on rate `tau`, which is much less than one.
        self.update_target(
            self.target_actor.variables, self.actor_model.variables, self.tauAC[0]
        )
        self.update_target(
            self.target_critic.variables, self.critic_model.variables, self.tauAC[1]
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
    def get_actor(
        self,
        num_states: int,
        num_actions: int,
        num_hidden: int = 256,
        num_layers: int = 2,
        action_bias: float = 0,
    ):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(num_states,))
        # dummy rescale to avoid recursive using of inputs, also placeholder for rescaling

        x = layers.Dense(
            num_hidden,
            activation="relu",
            kernel_initializer=tf.keras.initializers.HeNormal(),
        )(inputs)

        # if n_layers <= 1, the loop will be skipped in default
        for i in range(num_layers - 1):
            x = layers.Dense(
                num_hidden,
                activation="relu",
                kernel_initializer=tf.keras.initializers.HeNormal(),
            )(x)

        # output layer
        out = layers.Dense(
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
        eager_model = tf.keras.Model(inputs, out)
        # graph_model = tf.function(eager_model)
        return eager_model

    def get_critic(
        self,
        num_states: int,
        num_actions: int,
        num_hidden0: int = 16,
        num_hidden1: int = 32,
        num_hidden2: int = 256,
        num_layers: int = 2,
    ):
        # State as input
        state_input = layers.Input(shape=(num_states,))
        state_out = layers.Dense(num_hidden0, activation="relu")(state_input)
        state_out = layers.Dense(num_hidden1, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(
            shape=(num_actions,)
        )  # action is defined as flattened.
        action_out = layers.Dense(num_hidden1, activation="relu")(action_input)

        # Both are passed through separate layer before concatenating
        x = layers.Concatenate()([state_out, action_out])

        # if n_layers <= 1, the loop will be skipped in default
        for i in range(num_layers - 1):
            x = layers.Dense(
                num_hidden2,
                activation="relu",
                kernel_initializer=tf.keras.initializers.HeNormal(),
            )(x)
        x = layers.Dense(
            num_hidden2,
            activation="relu",
            kernel_initializer=tf.keras.initializers.HeNormal(),
        )(x)

        outputs = layers.Dense(1, activation=None)(x)

        # Outputs single value for give state-action
        eager_model = tf.keras.Model([state_input, action_input], outputs)
        # graph_model = tf.function(eager_model)

        return eager_model

    def start_episode(self, dt: datetime):
        self.logger.info(f"Episode start at {dt}", extra=dictLogger)
        # somehow mongodb does not like microseconds in rec['plot']
        dt_milliseconds = int(dt.microsecond / 1000) * 1000
        self.episode_start_dt = dt.replace(microsecond=dt_milliseconds)

    """
    `policy()` returns an action sampled from our Actor network plus some noise for
    exploration.
    """

    # action outputs and noise object are all row vectors of length 21*17 (r*c), output numpy array
    def policy(self, state):
        # We make sure action is within bounds
        # legal_action = np.clip(sampled_actions, action_lower, action_upper)

        states = tf.expand_dims(state, 0)  # motion states is 30*3 matrix
        sampled_actions = self.infer_single_sample(states)
        # return np.squeeze(sampled_actions)  # ? might be unnecessary
        return sampled_actions + self.ou_noise()

    def actor_predict(self, state, t):
        """
        `actor_predict()` returns an action sampled from our Actor network without noise.
        add optional t just to have uniform interface with rdpg
        """
        tt = t  # ddpg is not sequential
        return self.policy(state)

    @tf.function
    def infer_single_sample(self, state):
        # logger.info(f"Tracing", extra=dictLogger)
        print("Tracing infer!")
        sampled_actions = tf.squeeze(self.actor_model(state))
        # Adding noise to action
        return sampled_actions

    def deposit(self, prev_ts, prev_o_t, prev_a_t, prev_table_start, cycle_reward, o_t):
        if self.db_server:
            rec = {
                "timestamp": datetime.fromtimestamp(
                    float(
                        prev_ts.numpy()[0]
                    )  # fromtimestamp need float, tf data precision set to float32
                ),  # from ms to s
                "plot": {
                    "character": self.truck.TruckName,
                    "driver": self.driver,
                    "when": self.episode_start_dt,
                    "tz": str(self.truck.tz),
                    "where": "campus",
                    "states": {
                        "velocity_unit": "kmph",
                        "thrust_unit": "percentage",
                        "brake_unit": "percentage",
                        "length": o_t.shape[0],
                    },
                    "actions": {
                        "action_row_number": self.truck.ActionFlashRow,
                        "action_column_number": self.truck.PedalScale,
                    },
                    "reward": {
                        "reward_unit": "wh",
                    },
                },
                "observation": {
                    "state": prev_o_t.numpy().tolist(),
                    "action": prev_a_t.numpy().tolist(),
                    "action_start_row": prev_table_start,
                    "reward": cycle_reward.numpy().tolist(),
                    "next_state": o_t.numpy().tolist(),
                },
            }
            self.buffer.store_db(rec)
        else:
            self.buffer.store_npa(
                (
                    prev_o_t,
                    prev_a_t,
                    cycle_reward,
                    o_t,
                )
            )

    def end_episode(self):
        self.logger.info(f"Episode end at {datetime.now()}", extra=dictLogger)

    def touch_gpu(self):
        # tf.summary.trace_on(graph=True, profiler=True)
        # ignites manual loading of tensorflow library, to guarantee the real-time processing of first data in main thread
        init_states = tf.random.normal(
            (self.num_states,)
        )  # state must have 30*5 (speed, throttle, current, voltage) 5 tuple

        action0 = self.policy(init_states)
        self.logger.info(
            f"manual load tf library by calling convert_to_tensor",
            extra=dictLogger,
        )
        self.ou_noise.reset()

        # warm up gpu training graph execution pipeline
        if self.buffer.buffer_counter != 0:
            if not self.infer:
                self.logger.info(
                    f"ddpg warm up training!",
                    extra=dictLogger,
                )

                (_, _) = self.train()
                self.update_target(
                    self.target_actor.variables,
                    self.actor_model.variables,
                    self.tauAC[0],
                )
                # self.logger.info(f"Updated target actor", extra=self.dictLogger)
                self.update_target(
                    self.target_critic.variables,
                    self.critic_model.variables,
                    self.tauAC[1],
                )

                # self.logger.info(f"Updated target critic.", extra=self.dictLogger)
                self.logger.info(
                    f"ddpg warm up training done!",
                    extra=dictLogger,
                )


    def train(self):

        state_batch, action_batch, reward_batch, next_state_batch = self.buffer.sample_minibatch_ddpg()
        critic_loss, actor_loss = self.update_with_batch(state_batch, action_batch, reward_batch, next_state_batch)
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
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        print("Tracing update!")
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + self.gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
            )
            # ? need to confirm since replay buffer will take max over the actions of Q function.:with
            # future_rewards = self.target_critic(
            #             #     [next_state_batch, target_actions], training=True
            #             # )
            # y = reward_batch + self.gamma * tf.reduce_max(future_rewards, axis = 1)
            # ! the question above is not necessary, since deterministic policy is the maximum!
            critic_value = self.critic_model([state_batch, action_batch], training=True)
            # scalar value, average over the batch
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        # logger.info(f"BP done.", extra=dictLogger)

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            # scalar value, average over the batch
            actor_loss = -tf.math.reduce_mean(critic_value)

        # gradient director directly over actor model weights
        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        # TODO Check if this is correct. compare above actor_grad tensor with below
        # action_gradients= tape.gradient(actions, actor_model.trainable_variables)
        # actor_grad = tape.gradient(actor_loss, actions, action_gradients)

        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )
        return critic_loss, actor_loss

    # we only calculate the loss
    @tf.function
    def infer_with_batch(
            self,
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        target_actions = self.target_actor(next_state_batch, training=True)
        y = reward_batch + self.gamma * self.target_critic(
            [next_state_batch, target_actions], training=True
        )
        critic_value = self.critic_model([state_batch, action_batch], training=True)
        critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        self.logger.info(f"No update Calulate reward done.", extra=dictLogger)

        actions = self.actor_model(state_batch, training=True)
        critic_value = self.critic_model([state_batch, actions], training=True)
        # Used `-value` as we want to maximize the value given
        # by the critic for our actions
        actor_loss = -tf.math.reduce_mean(critic_value)

        return critic_loss, actor_loss


    # We only compute the loss and don't update parameters
    def get_losses(self):

        state_batch, action_batch, reward_batch, next_state_batch = self.buffer.sample_minibatch_ddpg()
        critic_loss, actor_loss = self.infer_with_batch(
            state_batch, action_batch, reward_batch, next_state_batch
        )
        return critic_loss, actor_loss

"""
## Training hyperparameters
"""

# std_dev = 0.2
# ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

# actor_model = get_actor()
# critic_model = get_critic()

# target_actor = get_actor()
# target_critic = get_critic()
#
# # Making the weights equal initially
# target_actor.set_weights(actor_model.get_weights())
# target_critic.set_weights(critic_model.get_weights())

# # Learning rate for actor-critic models
# critic_lr = 0.002
# actor_lr = 0.001
#
# critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
# actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

# total_episodes = 100
# # Discount factor for future rewards
# gamma = 0.99
# # Used to update target networks
# tau = 0.005
#
# buffer = Buffer(50000, 64)

"""
Now we implement our main training loop, and iterate over episodes.
We sample actions using `policy()` and train with `learn()` at each time step,
along with updating the Target networks at a rate `tau`.
"""
#
# # To store reward history of each episode
# ep_reward_list = []
# # To store average reward history of last few episodes
# avg_reward_list = []
#
# # Takes about 4 min to train
# for ep in range(total_episodes):
#
#     prev_state = env.reset()
#     episodic_reward = 0
#
#     while True:
#         # Uncomment this to see the Actor in action
#         # But not in a python notebook.
#         # env.render()
#
#         tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
#
#         action = policy(tf_prev_state, ou_noise)
#         # Recieve state and reward from environment.
#         state, reward, done, info = env.step(action)
#
#         buffer.record((prev_state, action, reward, state))
#         episodic_reward += reward
#
#         buffer.learn()
#         update_target(target_actor.variables, actor_model.variables, tau)
#         update_target(target_critic.variables, critic_model.variables, tau)
#
#         # End this episode when `done` is True
#         if done:
#             break
#
#         prev_state = state
#
#     ep_reward_list.append(episodic_reward)
#
#     # Mean of last 40 episodes
#     avg_reward = np.mean(ep_reward_list[-40:])
#     print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
#     avg_reward_list.append(avg_reward)
#
# # Plotting graph
# # Episodes versus Avg. Rewards
# plt.plot(avg_reward_list)
# plt.xlabel("Episode")
# plt.ylabel("Avg. Epsiodic Reward")
# plt.show()

"""
If training proceeds correctly, the average episodic reward will increase with time.

Feel free to try different learning rates, `tau` values, and architectures for the
Actor and Critic networks.

The Inverted Pendulum problem has low complexity, but DDPG work great on many other
problems.

Another great environment to try this on is `LunarLandingContinuous-v2`, but it will take
more episodes to obtain good results.
"""

# # Save the weights
# actor_model.save_weights("pendulum_actor.h5")
# critic_model.save_weights("pendulum_critic.h5")
#
# target_actor.save_weights("pendulum_target_actor.h5")
# target_critic.save_weights("pendulum_target_critic.h5")

"""
Before Training:

![before_img](https://i.imgur.com/ox6b9rC.gif)
"""

"""
After 100 episodes:

![after_img](https://i.imgur.com/eEH8Cz6.gif)
"""
