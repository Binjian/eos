# third-party imports
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.initializers as initializers

# local imports
from ...l045a_rdpg import logger, logc, logd, dictLogger
from ..utils.ou_noise import OUActionNoise


class ActorNet:
    """Actor network for the RDPG algorithm."""

    def __init__(
        self,
        state_dim,
        action_dim,
        sequence_len,
        batch_size,
        hidden_dim,
        n_layers,
        padding_value,
        gamma,
        tau,
        lr,
    ):
        """Initialize the actor network.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Dimension of the hidden layer.
            lr (float): Learning rate for the network.
        """

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.sequence_len = sequence_len
        self.lr = lr
        self.padding_value = padding_value
        self.n_layers = n_layers
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        inputs = layers.Input(shape=(sequence_len, state_dim))

        # attach mask to the inputs, & apply recursive lstm layer to the output
        x = layers.Masking(mask_value=self.padding_value)(
            inputs
        )  # input (observation) padded with -10000.0

        # dummy rescale to avoid recursive using of inputs, also placeholder for rescaling
        # x = layers.Rescale(inputs, 1.0)

        # if n_layers <= 1, the loop will be skipped in default
        for i in range(n_layers - 1):
            x = layers.LSTM(hidden_dim, return_sequences=True)(x)

        lstm_outputs = layers.LSTM(
            hidden_dim, return_sequences=True, return_state=False
        )(x)

        # rescale the output of the lstm layer to (-1, 1)
        action_outputs = layers.Dense(action_dim, activation="tanh")(lstm_outputs)

        self.eager_model = tf.keras.Model(inputs, action_outputs)
        # no need to evaluate the last action separately
        # just run the model inference and get the last action
        # self.action_last = action_outputs[:, -1, :]

        self.eager_model.summary()
        # self.graph_model = tf.function(self.eager_model)
        self.optimizer = tf.keras.optimizers.Adam(lr)

        std_dev = 0.2
        self.ou_noise = OUActionNoise(
            mean=np.zeros(action_dim),
            std_deviation=float(std_dev) * np.ones(action_dim),
        )

    def predict(self, state):
        """Predict the action given the state.
        Args:
            state (np.array): State, Batch dimension needs to be one.

        Returns:
            np.array: Action
        """
        # logc("ActorNet.predict")
        action_seq = self.eager_model(state)

        # get the last step action and squeeze the batch dimension
        last_action = tf.squeeze(action_seq[:, -1, :])
        sampled_action = last_action + self.ou_noise()  # noise object is a row vector
        # logc("ActorNet.predict")
        return sampled_action

    def clone_weights(self, target):
        """Clone weights from a model to another model. only for target critic"""
        self.eager_model.set_weights(target.eager_model.get_weights())

    def soft_update(self, target):
        """Update the target critic weights. only for target critic."""
        self.eager_model.set_weights(
            [
                self.tau * w + (1 - self.tau) * w_t
                for w, w_t in zip(
                    self.eager_model.get_weights(), target.eager_model.get_weights()
                )
            ]
        )

    def evaluate_actions(self, state):
        """Evaluate the action given the state.
        Args:
            state (np.array): State, in a minibatch

        Returns:
            np.array: Q-value
        """
        # logc("ActorNet.evaluate_actions")
        return self.eager_model(state)
