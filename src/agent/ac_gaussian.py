import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.initializers as initializers
import tensorflow_probability as tfp

""" one training step to update network"""
def train_step(net, history, optimizer, tape):
    huber_loss = keras.losses.Huber()
    actor_losses = []
    critic_losses = []
    for action, mu_sigma, value, ret in history:
        # at this point in history, the critic estimated that we would get a
        # total reward = `value` in the future. we took an action with log probability
        # of `log_prob` and ended up recieving a total reward = `ret`.
        # the actor must be updated so that it predicts an action that leads to
        # high rewards (compared to critic's estimate) with high probability.
        diff = ret - value
        actor_loss = customlossgaussian(mu_sigma, action, diff)
        actor_losses.append(actor_loss)  # actor loss

        # the critic must be updated so that it predicts a better estimate of
        # the future rewards.
        # todo calculate loss_critic
        critic_losses.append(
            huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
        )

    # now the agent backpropagate every episode. todo or backpropagation every n (say 20) episodes
    # backpropagation
    loss_value = sum(actor_losses) + 300 * sum(critic_losses)  # todo 300-400 times

    grads = tape.gradient(loss_value, net.trainable_variables)
    optimizer.apply_gradients(
        zip(grads, net.trainable_variables)
    )
    return loss_value



"""construct the actor network with mu and sigma as output"""
def constructactorcriticnetwork(
        num_observations, sequence_len, num_actions, num_hidden, bias_mu, bias_sigma
):
    inputs = layers.Input(
        shape=(sequence_len, num_observations)  # DONE should be flattened
    )  # input dimension, 3 rows, 20 columns.
    # add flatten layer
    flatinputs = layers.Flatten()(inputs)
    hidden = layers.Dense(
        num_hidden, activation="relu", kernel_initializer=initializers.he_normal()
    )(flatinputs)
    common = layers.Dense(
        num_hidden, activation="relu", kernel_initializer=initializers.he_normal()
    )(hidden)
    mu = layers.Dense(
        num_actions,
        activation="tanh",  # tanh for mu between (-1,+1) to be scaled by a hyperparameter
        # activation="linear",
        kernel_initializer=initializers.zeros(),
        bias_initializer=initializers.constant(bias_mu),
    )(common)
    sigma = layers.Dense(
        num_actions,
        # activation="sigmoid", # optional to use sigmoid to get bounded sigma (0,1)
        activation="softplus",  # use softplus to ensure positive sigma
        kernel_initializer=initializers.zeros(),
        bias_initializer=initializers.constant(bias_sigma),
    )(common)

    critic_value = layers.Dense(1)(common)

    # sigma1 = sigma * 0.1 # first try using softplus without bound; next to try sigmoid with coefficent
    mu_sigma = tf.stack([mu, sigma])
    actorcritic_network = keras.Model(inputs=inputs, outputs=[mu_sigma, critic_value])

    return actorcritic_network


"""
weighted gaussian log likelihood loss function at time t
modeling the multivariate normal distribution as independent in all dimensions, ???
so that the logit are summed up in all dimensions.
for episode calculation needs to store the history and call the function with history batch data.

"""


def customlossgaussian(mu_sigma, action, reward):
    # obtain mu and sigma from actor network
    nn_mu, nn_sigma = tf.unstack(mu_sigma)

    # obtain pdf of gaussian distribution
    pdf_value = (
        tf.exp(-0.5 * ((action - nn_mu) / (nn_sigma)) ** 2)
        * 1
        / (nn_sigma * tf.sqrt(2 * np.float64(np.pi)))
    )

    # compute log probability
    log_probability = tf.math.log(pdf_value + 1e-5)

    # add up all dimenstions
    log_probability_sum = tf.math.reduce_sum(log_probability)
    # compute weighted loss
    # actor loss shoud be negative, negative loss is the sum of reward
    loss_actor = -reward * log_probability_sum
    # add entropy loss (Gaussian Entropy) to reduce randomness with training onging
    # entropy loss shoud be positive
    k_loss = 1.0 # todo too small DONE adjust to a bigger value original value 1e-4;
    loss_entropy = k_loss * (
        tf.math.log(2 * np.float64(np.pi) * tf.math.square(tf.norm(nn_sigma)) + 1) / 2
    )

    loss = loss_actor + loss_entropy
    return loss