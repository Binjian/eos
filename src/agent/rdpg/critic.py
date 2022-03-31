# third-party imports
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.initializers as initializers

# local imports
from ...l045a_rdpg import logger, logc, logd, dictLogger


class CriticNet:
    """Critic network for the RDPG algorithm."""

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
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lr = lr
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.padding_value = padding_value

        inputs_state = layers.Input(shape=(sequence_len, state_dim))
        inputs_action = layers.Input(shape=(sequence_len, action_dim))
        # concatenate state and action along the feature dimension
        # both state and action are from padded minibatch, only for training
        inputs_state_action = layers.concatenate(-1)([inputs_state, inputs_action])

        # attach mask to the inputs, & apply recursive lstm layer to the output
        x = layers.Masking(mask_value=self.padding_value)(
            inputs_state_action
        )  # input (observation) padded with -10000.0

        # if n_layers <= 1, the loop will be skipped in default
        for i in range(n_layers - 1):
            x = layers.LSTM(hidden_dim, return_sequences=True)(x)

        lstm_outputs = layers.LSTM(
            hidden_dim, return_sequences=True, return_state=False
        )(x)

        critic_outputs = layers.Dense(1, activation=None)(lstm_outputs)

        self.eager_model = tf.keras.Model(
            inputs=[inputs_state, inputs_action], outputs=critic_outputs
        )

        self.eager_model.summary()
        # self.graph_model = tf.function(self.eager_model)
        self.optimizer = tf.keras.optimizers.Adam(lr)

    def clone_weights(self, target):
        """Clone weights from a model to another model."""
        self.eager_model.set_weights(target.eager_model.get_weights())

    def soft_update(self, target):
        """Update the target critic weights."""
        self.eager_model.set_weights(
            [
                self.tau * w + (1 - self.tau) * w_t
                for w, w_t in zip(
                    self.eager_model.get_weights(), target.eager_model.get_weights()
                )
            ]
        )

    def evaluate_q(self, state, action):
        """Evaluate the action value given the state and action
        Args:
            state (np.array): State in a minibatch
            action (np.array): Action in a minibatch

        Returns:
            np.array: Q-value
        """
        # logc("ActorNet.evaluate_actions")
        return self.eager_model(state, action)
