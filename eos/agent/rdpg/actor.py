# third-party imports
import numpy as np
import tensorflow as tf
from keras import layers
import keras.initializers as initializers
from eos import logger, dictLogger


# local imports
from ..utils.ou_noise import OUActionNoise
from eos.utils.exception import ReadOnlyError


class ActorNet:
    """Actor network for the RDPG algorithm."""

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim,
        n_layers,
        padding_value,
        tau,
        lr,
        ckpt_dir,
        ckpt_interval,
    ):
        """Initialize the actor network.

        restore checkpoint from the provided directory if it exists,
        initialize otherwise.
        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Dimension of the hidden layer.
            lr (float): Learning rate for the network.
            ckpt_dir (str): Directory to restore the checkpoint from.
        """

        self._state_dim = state_dim
        self._action_dim = action_dim
        self._hidden_dim = hidden_dim
        self._lr = lr
        self._padding_value = padding_value
        self._n_layers = n_layers
        self._tau = tau

        inputs = layers.Input(shape=(None, state_dim))

        # attach mask to the inputs, & apply recursive lstm layer to the output
        x = layers.Masking(mask_value=padding_value)(
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

        self._std_dev = 0.2
        self.ou_noise = OUActionNoise(
            mean=np.zeros(action_dim),
            std_deviation=float(self._std_dev) * np.ones(action_dim),
        )

        # restore the checkpoint if it exists
        self.ckpt_dir = ckpt_dir
        self._ckpt_interval = ckpt_interval
        self.ckpt = tf.train.Checkpoint(
            step=tf.Variable(1), optimizer=self.optimizer, net=self.eager_model
        )
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, self.ckpt_dir, max_to_keep=10
        )
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            logger.info(
                f"Restored actor from {self.ckpt_manager.latest_checkpoint}",
                extra=dictLogger,
            )
        else:
            logger.info(f"Actor Initializing from scratch", extra=dictLogger)

    def clone_weights(self, moving_net):
        """Clone weights from a model to another model. only for target critic"""
        self.eager_model.set_weights(moving_net.eager_model.get_weights())

    def soft_update(self, moving_net):
        """Update the target critic weights. only for target critic."""
        self.eager_model.set_weights(
            [
                self._tau * w + (1 - self._tau) * w_t
                for w, w_t in zip(
                    moving_net.eager_model.get_weights(), self.eager_model.get_weights()
                )
            ]
        )

    def save_ckpt(self):
        self.ckpt.step.assign_add(1)
        if int(self.ckpt.step) % self.ckpt_interval == 0:
            save_path = self.ckpt_manager.save()
            logger.info(
                f"Saved ckpt for step {int(self.ckpt.step)}: {save_path}",
                extra=dictLogger,
            )

    def reset_noise(self):
        self.ou_noise.reset()

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

    def evaluate_actions(self, state):
        """Evaluate the action given the state.
        Args:
            state (np.array): State, in a minibatch

        Returns:
            np.array: Q-value
        """
        # logc("ActorNet.evaluate_actions")
        return self.eager_model(state)

    @property
    def state_dim(self):
        return self._state_dim

    @state_dim.setter
    def state_dim(self, value):
        raise ReadOnlyError("state_dim is read-only")

    @property
    def action_dim(self):
        return self._action_dim

    @action_dim.setter
    def action_dim(self, value):
        raise ReadOnlyError("action_dim is read-only")

    @property
    def hidden_dim(self):
        return self._hidden_dim

    @hidden_dim.setter
    def hidden_dim(self, value):
        raise ReadOnlyError("hidden_dim is read-only")

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, value):
        raise ReadOnlyError("lr is read-only")

    @property
    def padding_value(self):
        return self._padding_value

    @padding_value.setter
    def padding_value(self, value):
        raise ReadOnlyError("padding_value is read-only")

    @property
    def n_layers(self):
        return self._n_layers

    @n_layers.setter
    def n_layers(self, value):
        raise ReadOnlyError("n_layers is read-only")

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, value):
        raise ReadOnlyError("tau is read-only")

    @property
    def ckpt_interval(self):
        return self._ckpt_interval

    @ckpt_interval.setter
    def ckpt_interval(self, value):
        raise ReadOnlyError("ckpt_interval is read-only")
