# third-party imports
import tensorflow as tf
from tensorflow.keras import layers

# local imports
from eos import dictLogger, logger
from eos.utils.exception import ReadOnlyError


class CriticNet:
    """Critic network for the RDPG algorithm."""

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
        """Initialize the critic network.

                restore checkpoint from the provided directory if it exists,
                initialize otherwise.
                Args:
                    state_dim (int): Dimension of the state space.
                    action_dim (int): Dimension of the action space.
                    hidden_dim (int): Dimension of the hidden layer.
                    lr (float): Learning rate for the network.
                    ckpt_dir (str): Directory to restore the checkpoint from.
        i"""

        self._state_dim = state_dim
        self._action_dim = action_dim
        self._hidden_dim = hidden_dim
        self._n_layers = n_layers
        self._lr = lr
        self._tau = tau
        self._padding_value = padding_value

        inputs_state = layers.Input(shape=(None, state_dim))
        inputs_action = layers.Input(shape=(None, action_dim))
        # concatenate state and action along the feature dimension
        # both state and action are from padded minibatch, only for training
        inputs_state_action = layers.Concatenate(axis=-1)([inputs_state, inputs_action])

        # attach mask to the inputs, & apply recursive lstm layer to the output
        x = layers.Masking(mask_value=self.padding_value)(
            inputs_state_action
        )  # input (observation) padded with -10000.0

        # if n_layers <= 1, the loop will be skipped in default
        for i in range(n_layers - 1):
            x = layers.LSTM(hidden_dim, return_sequences=True, return_state=False)(x)

        lstm_output = layers.LSTM(
            hidden_dim, return_sequences=False, return_state=False
        )(x)

        critic_output = layers.Dense(1, activation=None)(lstm_output)

        self.eager_model = tf.keras.Model(
            inputs=[inputs_state, inputs_action], outputs=critic_output
        )

        self.eager_model.summary()
        # self.graph_model = tf.function(self.eager_model)
        self.optimizer = tf.keras.optimizers.Adam(lr)

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
                f'Restored actor from {self.ckpt_manager.latest_checkpoint}',
                extra=dictLogger,
            )
        else:
            logger.info(f'Critic Initializing from scratch', extra=dictLogger)

    def clone_weights(self, moving_net):
        """Clone weights from a model to another model."""
        self.eager_model.set_weights(moving_net.eager_model.get_weights())

    def soft_update(self, moving_net):
        """Update the target critic weights."""
        self.eager_model.set_weights(
            [
                self.tau * w + (1 - self.tau) * w_t
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
                f'Saved ckpt for step {int(self.ckpt.step)}: {save_path}',
                extra=dictLogger,
            )

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, 600], dtype=tf.float32),
            tf.TensorSpec(shape=[None, None, 68], dtype=tf.float32),
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
        return self.eager_model([state, action])

    @property
    def state_dim(self):
        return self._state_dim

    @state_dim.setter
    def state_dim(self, value):
        raise ReadOnlyError('state_dim is read-only')

    @property
    def action_dim(self):
        return self._action_dim

    @action_dim.setter
    def action_dim(self, value):
        raise ReadOnlyError('action_dim is read-only')

    @property
    def hidden_dim(self):
        return self._hidden_dim

    @hidden_dim.setter
    def hidden_dim(self, value):
        raise ReadOnlyError('hidden_dim is read-only')

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, value):
        raise ReadOnlyError('lr is read-only')

    @property
    def padding_value(self):
        return self._padding_value

    @padding_value.setter
    def padding_value(self, value):
        raise ReadOnlyError('padding_value is read-only')

    @property
    def n_layers(self):
        return self._n_layers

    @n_layers.setter
    def n_layers(self, value):
        raise ReadOnlyError('n_layers is read-only')

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, value):
        raise ReadOnlyError('tau is read-only')

    @property
    def ckpt_interval(self):
        return self._ckpt_interval

    @ckpt_interval.setter
    def ckpt_interval(self, value):
        raise ReadOnlyError('ckpt_interval is read-only')
