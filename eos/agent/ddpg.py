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
import numpy as np
import tensorflow as tf
from keras import layers
import keras.initializers as initializers

from eos import logger, dictLogger

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


class Buffer:
    def __init__(
        self,
        actor_model,
        critic_model,
        target_actor,
        target_critic,
        actor_optimizer,
        critic_optimizer,
        num_observations,
        sequence_len,
        num_actions,
        buffer_capacity=10000,
        batch_size=4,
        gamma=0.99,
        load_buffer=False,
        file_sb=None,
        file_ab=None,
        file_rb=None,
        file_nsb=None,
        file_bc=None,
        datafolder="./",
    ):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.num_observations = num_observations
        self.sequence_len = sequence_len
        self.num_actions = num_actions
        self.data_folder = datafolder
        self.file_sb = self.data_folder + "/state_buffer.npy"
        self.file_ab = self.data_folder + "/action_buffer.npy"
        self.file_rb = self.data_folder + "/reward_buffer.npy"
        self.file_nsb = self.data_folder + "/next_state_buffer.npy"
        self.file_bc = self.data_folder + "/buffer_counter.npy"
        self.state_buffer = None
        self.action_buffer = None
        self.reward_buffer = None
        self.next_state_buffer = None
        # Its tells us num of times record() was called.
        self.buffer_counter = 0
        self.load()

        self.actor_model = actor_model
        self.critic_model = critic_model
        self.target_actor = target_actor
        self.target_critic = target_critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.gamma = gamma

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    def save(self):

        np.save(self.file_sb, self.state_buffer)
        np.save(self.file_ab, self.action_buffer)
        np.save(self.file_rb, self.reward_buffer)
        np.save(self.file_nsb, self.next_state_buffer)
        np.save(self.file_bc, self.buffer_counter)
        print(f"saved buffer counter: {self.buffer_counter}")

    def load_default(self):

        try:
            self.state_buffer = np.load(self.file_sb)
            self.action_buffer = np.load(self.file_ab)
            self.reward_buffer = np.load(self.file_rb)
            self.next_state_buffer = np.load(self.file_nsb)
            self.buffer_counter = np.load(self.file_bc)
            print("load last default experience")
            print(f"loaded default buffer counter: {self.buffer_counter}")
        except IOError:
            self.state_buffer = np.zeros(
                (self.buffer_capacity, self.sequence_len, self.num_observations)
            )
            self.action_buffer = np.zeros((self.buffer_capacity, self.num_actions))
            self.reward_buffer = np.zeros((self.buffer_capacity, 1))
            self.next_state_buffer = np.zeros(
                (self.buffer_capacity, self.sequence_len, self.num_observations)
            )
            self.buffer_counter = 0
            print("blank experience")

    def load(self):
        if (
            (not self.file_sb)
            or (not self.file_ab)
            or (not self.file_rb)
            or (not self.file_nsb)
            or (not self.file_bc)
        ):
            self.load_default()
        else:
            try:
                self.state_buffer = np.load(self.file_sb)
                self.action_buffer = np.load(self.file_ab)
                self.reward_buffer = np.load(self.file_rb)
                self.next_state_buffer = np.load(self.file_nsb)
                self.buffer_counter = np.load(self.file_bc)
                print("load last specified experience")
                print(f"loaded buffer counter: {self.buffer_counter}")
            except IOError:
                self.load_default()

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self,
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
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

        logger.info(f"BP done.", extra=dictLogger)

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

    # We compute the loss and update parameters
    def learn(self):
        # get sampling range, if not enough data, batch is small,
        # batch size starting from 1, until reach buffer
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # randomly sample indices , in case batch_size > record_range, numpy default is repeated samples
        batch_indices = np.random.choice(record_range, self.batch_size)

        # convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        critic_loss, actor_loss = self.update(
            state_batch, action_batch, reward_batch, next_state_batch
        )
        return critic_loss, actor_loss

    # we only calculate the loss
    @tf.function
    def noupdate(
        self,
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
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
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        logger.info(f"No update Calulate reward done.", extra=dictLogger)

        # critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        # self.critic_optimizer.apply_gradients(
        #     zip(critic_grad, self.critic_model.trainable_variables)
        # )

        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        # actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        # self.actor_optimizer.apply_gradients(
        #     zip(actor_grad, self.actor_model.trainable_variables)
        # )
        return critic_loss, actor_loss

    # We only compute the loss and don't update parameters
    def nolearn(self):
        # get sampling range, if not enough data, batch is small,
        # batch size starting from 1, until reach buffer
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # randomly sample indices , in case batch_size > record_range, numpy default is repeated samples
        batch_indices = np.random.choice(record_range, self.batch_size)

        # convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        critic_loss, actor_loss = self.noupdate(
            state_batch, action_batch, reward_batch, next_state_batch
        )
        return critic_loss, actor_loss


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


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
    num_observations,
    num_actions,
    sequence_len,
    num_hidden,
    action_bias,
):
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(sequence_len, num_observations))
    flatinputs = layers.Flatten()(inputs)
    hidden = layers.Dense(
        num_hidden, activation="relu", kernel_initializer=initializers.he_normal()
    )(flatinputs)
    hidden1 = layers.Dense(
        num_hidden, activation="relu", kernel_initializer=initializers.he_normal()
    )(hidden)
    out = layers.Dense(
        num_actions,
        activation="tanh",
        kernel_initializer=last_init,
        bias_initializer=initializers.constant(action_bias),
    )(hidden1)

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
    dim_observations,
    dim_actions,
    sequence_len,
    num_hidden0=16,
    num_hidden1=32,
    num_hidden2=256,
):
    # State as input
    state_input = layers.Input(shape=(sequence_len, dim_observations))
    state_flattened = layers.Flatten()(state_input)
    state_out = layers.Dense(num_hidden0, activation="relu")(state_flattened)
    state_out = layers.Dense(num_hidden1, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(dim_actions,))  # action is defined as flattened.
    action_out = layers.Dense(num_hidden1, activation="relu")(action_input)

    # Both are passed through separate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(num_hidden2, activation="relu")(concat)
    out = layers.Dense(num_hidden2, activation="relu")(out)
    outputs = layers.Dense(1, activation=None)(out)

    # Outputs single value for give state-action
    eager_model = tf.keras.Model([state_input, action_input], outputs)
    # graph_model = tf.function(eager_model)

    return eager_model


"""
`policy()` returns an action sampled from our Actor network plus some noise for
exploration.
"""

# action outputs and noise object are all row vectors of length 21*17 (r*c), output numpy array
def policy(actor_model, state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state)).numpy()
    noise = noise_object()  # noise object is a row vector
    # Adding noise to action
    sampled_actions = sampled_actions + noise

    # We make sure action is within bounds
    # legal_action = np.clip(sampled_actions, action_lower, action_upper)

    # return np.squeeze(sampled_actions)  # ? might be unnecessary
    return sampled_actions


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
