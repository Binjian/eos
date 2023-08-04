# third-party imports
from typing import ClassVar
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers

from pathlib import Path
from eos.utils import dictLogger, logger
from eos.utils.exception import ReadOnlyError

# local imports
from eos.agent.utils.ou_noise import OUActionNoise
from eos.agent.utils.hyperparams import HyperParamRDPG


class ActorNet:
    """Actor network for the RDPG algorithm."""

    _hyperparams: ClassVar[
        HyperParamRDPG
    ] = HyperParamRDPG()  # for tf.function to get some of the default hyperparameters

    def __init__(
        self,
        state_dim: int = 0,
        action_dim: int = 0,
        hidden_dim: int = 0,
        n_layers: int = 0,
        padding_value: float = 0.0,
        tau: float = 0.0,
        lr: float = 0.0,
        ckpt_dir: Path = Path("."),
        ckpt_interval: int = 0,
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

        states = layers.Input(shape=(None, state_dim))
        last_actions = layers.Input(shape=(None, action_dim))

        inputs = [
            states,
            last_actions,
        ]  # history update consists of current states s_t and last actions a_{t-1}
        x = layers.Concatenate(axis=-1)(
            inputs
        )  # feature dimension would be [states + actions]

        # attach mask to the inputs, & apply recursive lstm layer to the output
        x = layers.Masking(mask_value=padding_value)(
            x
        )  # input (observation) padded with -10000.0, on the time dimension

        x = layers.Dense(hidden_dim, activation="relu")(
            x
        )  # linear layer to map [states + actions] to [hidden_dim]

        # if n_layers <= 1, the loop will be skipped in default
        for i in range(n_layers - 1):
            x = layers.LSTM(
                hidden_dim,
                return_sequences=True,
                return_state=False,
                stateful=True,  # stateful for batches of long sequences split into batches of shorter sequences
            )(
                x
            )  # only return full sequences of hidden states, necessary for stacking LSTM layers,
            # last hidden state is not needed

        lstm_output = layers.LSTM(
            hidden_dim,
            return_sequences=False,
            return_state=False,  # return hidden and cell states for inference of each time step,
            stateful=True,  # stateful for batches of long sequences split into batches of shorter sequences
            # need to reset_states when the episode ends
        )(x)

        # rescale the output of the lstm layer to (-1, 1)
        action_output = layers.Dense(action_dim, activation="tanh")(lstm_output)

        self.eager_model = tf.keras.Model([states, last_actions], action_output)
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
            step=tf.Variable(tf.constant(1)),
            optimizer=self.optimizer,
            net=self.eager_model,
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
                    moving_net.eager_model.get_weights(),
                    self.eager_model.get_weights(),
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

    # @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, self._state_dim], dtype=tf.float32)])
    def predict(self, states: tf.Tensor, last_actions: tf.Tensor) -> tf.Tensor:
        """Predict the action given the state. Batch dimension needs to be one.
        Args:
            states: State, Batch dimension needs to be one.
            last_actions: Last action, Batch dimension needs to be one.

        Returns:
            Action
        """

        # get the last step action and squeeze the batch dimension
        action = self.predict_step(states, last_actions)
        sampled_action = (
            action + self.ou_noise()
        )  # noise object is a row vector, without batch and time dimension
        return sampled_action

    @tf.function(
        input_signature=[
            tf.TensorSpec(
                shape=[None, None, _hyperparams.NStates], dtype=tf.float32
            ),  # [None, None, 600] for cloud / [None, None, 90] for kvaser
            tf.TensorSpec(
                shape=[None, None, _hyperparams.NActions], dtype=tf.float32
            ),  # [None, None, 68] for both cloud and kvaser
        ]
    )
    def predict_step(self, states, last_actions):
        """Predict the action given the state.
        For Inferring
        Args:
            states (tf.Tensor): State, Batch dimension needs to be one.
            last_actions (tf.Tensor): State, Batch dimension needs to be one.

        Returns:
            np.array: Action, ditch the batch dimension
        """
        action_seq = self.eager_model([states, last_actions])

        # get the last step action and squeeze the time dimension,
        # since Batch is one when inferring, squeeze also the batch dimension by tf.squeeze default
        action = tf.squeeze(action_seq[:, -1, :])
        return action

    @tf.function(
        input_signature=[
            tf.TensorSpec(
                shape=[None, None, _hyperparams.NStates], dtype=tf.float32
            ),  # [None, None, 600] for cloud / [None, None, 90] for kvaser
            tf.TensorSpec(
                shape=[None, None, _hyperparams.NActions], dtype=tf.float32
            ),  # [None, None, 68] for both cloud and kvaser
        ]
    )
    def evaluate_actions(self, states, last_actions):
        """Evaluate the action given the state.
        For training
        Args:
            states (tf.Tensor): State, Batch dimension needs to be one.
            last_actions (tf.Tensor): State, Batch dimension needs to be one.

        Returns:
            np.array: Action, keep the batch dimension
        """
        return self.eager_model([states, last_actions])

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
