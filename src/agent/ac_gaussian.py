
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.initializers as initializers
import tensorflow_probability as tfp

"""construct the actor network with mu and sigma as output"""
def constructactorcriticnetwork(num_observations, sequence_len, num_actions, num_hidden, bias_mu, bias_sigma):
    inputs = layers.Input(
        shape=(sequence_len, num_observations) # TODO should be flattened
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
        activation="linear",
        kernel_initializer=initializers.zeros(),
        bias_initializer=initializers.constant(bias_mu),
    )(common)
    sigma = layers.Dense(
        num_actions,
        activation="sigmoid",
        # activation="softplus",
        kernel_initializer=initializers.zeros(),
        bias_initializer=initializers.constant(bias_sigma),
    )(common)

    critic_value = layers.Dense(1)(common)

    sigma1 = sigma * 0.1
    mu_sigma = tf.stack([mu, sigma1])
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
            / (nn_sigma * tf.sqrt(2 * np.pi))
    )

    # compute log probability
    log_probability = tf.math.log(pdf_value + 1e-5)

    # add up all dimenstions
    log_probability_sum = tf.math.reduce_sum(log_probability)
    # compute weighted loss
    loss_actor = -reward * log_probability_sum

    return loss_actor

