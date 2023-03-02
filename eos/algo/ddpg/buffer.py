
import numpy as np
import tensorflow as tf

from pymongoarrow.monkey import patch_all

patch_all()

from eos import Pool, dictLogger, logger
from eos.config import db_servers_by_name, db_servers_by_host, record_schemas

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
        truck,
        driver,
        actor_model,
        critic_model,
        target_actor,
        target_critic,
        actor_optimizer,
        critic_optimizer,
        num_states,
        num_actions,
        buffer_capacity=10000,
        batch_size=4,
        gamma=0.99,
        datafolder="./",
        db_server="mongo_local",
    ):

        self.logger = logger.getChild("main").getChild("ddpg").getChild("Buffer")
        self.logger.propagate = True
        # Number of "experiences" to store at max
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.truck = truck
        self.driver = driver
        self.num_states = num_states
        self.num_actions = num_actions
        self.data_folder = datafolder
        self.db_server = db_server
        if self.db_server:
            self.db = db_servers_by_name.get(self.db_server)
            if self.db is None:
                account_server = [s.split(":") for s in self.db_server.split("@")]
                flat_account_server = [s for l in account_server for s in l]
                assert (len(account_server) == 1 and len(flat_account_server) == 2) or (
                    len(account_server) == 2 and len(flat_account_server) == 4
                ), f"Wrong format for db server {self.db_server}!"
                if len(account_server) == 1:
                    self.db = db_servers_by_host.get(flat_account_server[0])
                    assert (
                        self.db is not None and self.db.Port == flat_account_server[1]
                    ), f"Config mismatch for db server {self.db_server}!"

                else:
                    self.db = db_servers_by_host.get(flat_account_server[2])
                    assert (
                        self.db is not None
                        and self.db.Port == flat_account_server[3]
                        and self.db.Username == flat_account_server[0]
                        and self.db.Password == flat_account_server[1]
                    ), f"Config mismatch for db server {self.db_server}!"
            self.logger.info(
                f"Using db server {self.db_server} for record replay buffer..."
            )
            self.db_schema = record_schemas["record_deep"]
            self.pool = Pool(
                url="mongodb://" + self.db.Host + ":" + self.db.Port,
                username=self.db.Username,
                password=self.db.Password,
                schema=self.db_schema.STRUCTURE,
                db_name=self.db.DatabaseName,
                coll_name=self.db.RecCollName,
                debug=False,
            )
            self.buffer_counter = self.pool.count_items(
                vehicle_id=self.truck.TruckName, driver_id=self.driver
            )
            self.logger.info(
                f"Connected to MongoDB {self.db.DatabaseName}, collection {self.db.RecCollName}, record number {self.buffer_counter}",
                extra=dictLogger,
            )
        else: #elif self.db_server is '':
            self.buffer_capacity = tf.convert_to_tensor(buffer_capacity, dtype=tf.int64)
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
            self.buffer_counter = tf.convert_to_tensor(0, dtype=tf.int64)
            self.load()

        self.actor_model = actor_model
        self.critic_model = critic_model
        self.target_actor = target_actor
        self.target_critic = target_critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.gamma = gamma

    def __del__(self):
        if self.db_server:
            # for database, exit needs drop interface.
            self.pool.drop_mongo()
        else:
            self.save_replay_buffer()

    def deposit(self, rec: dict):
        """
        Record a new experience in the pool (database).
        """
        result = self.pool.deposit_item(rec)
        assert result.acknowledged == True, "Record not deposited!"
        rec_inserted = self.pool.find_item(result.inserted_id)
        assert rec_inserted == rec, "Record inserted is not the same as the one inserted!"
        self.buffer_counter = self.pool.count_items(
            vehicle_id=self.truck.TruckName, driver_id=self.driver
        )
        self.logger.info(f"Pool has {self.buffer_counter} records", extra=dictLogger)

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple: tuple):
        """
        Record a new experience in the buffer (numpy arrays).

        Set index to zero if buffer_capacity is exceeded,
        replacing old records
        """
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    def save_replay_buffer(self):

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
                (self.buffer_capacity, self.num_states)
            )
            self.action_buffer = np.zeros((self.buffer_capacity, self.num_actions))
            self.reward_buffer = np.zeros((self.buffer_capacity, 1))
            self.next_state_buffer = np.zeros(
                (self.buffer_capacity, self.num_states)
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

    # We compute the loss and update parameters
    def learn(self):
        """
        Update the actor and critic networks using the sampled batch.
        """
        if self.db_server:
            # get sampling range, if not enough data, batch is small
            self.logger.info(
                f"start test_pool_sample of size {self.batch_size, self.truck.TruckName, self.driver}.",
                extra=dictLogger,
            )
            assert self.buffer_counter > 0, "pool is empty"
            batch = self.pool.sample_batch_items(
                batch_size=self.batch_size,
                vehicle_id=self.truck.TruckName,
                driver_id=self.driver,
            )
            assert (
                    len(batch) == self.batch_size
            ), f"sampled batch size {len(batch)} not match sample size {self.batch_size}"

            # convert to tensors
            state = [rec["observation"]["state"] for rec in batch]
            action = [rec["observation"]["action"] for rec in batch]
            reward = [rec["observation"]["reward"] for rec in batch]
            next_state = [rec["observation"]["next_state"] for rec in batch]

            # the shape of the tensor is the same as the buffer
            state_batch = tf.convert_to_tensor(np.array(state))
            action_batch = tf.convert_to_tensor(np.array(action))
            reward_batch = tf.convert_to_tensor(np.array(reward))
            reward_batch = tf.cast(reward_batch, dtype=tf.float32)
            next_state_batch = tf.convert_to_tensor(np.array(next_state))

        else:
            # get sampling range, if not enough data, batch is small,
            # batch size starting from 1, until reach buffer
            # logger.info(f"Tracing!", extra=dictLogger)
            record_range = tf.math.minimum(self.buffer_counter, self.buffer_capacity)
            # randomly sample indices , in case batch_size > record_range, numpy default is repeated samples
            batch_indices = np.random.choice(record_range, self.batch_size)

            # convert to tensors
            state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
            action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
            reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
            reward_batch = tf.cast(reward_batch, dtype=tf.float32)
            next_state_batch = tf.convert_to_tensor(
                self.next_state_buffer[batch_indices]
            )

        critic_loss, actor_loss = self.update(
            state_batch, action_batch, reward_batch, next_state_batch
        )
        return critic_loss, actor_loss

    # we only calculate the loss
    @tf.function
    def calc_losses(
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

        # critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        # self.critic_optimizer.apply_gradients(
        #     zip(critic_grad, self.critic_model.trainable_variables)
        # )

        actions = self.actor_model(state_batch, training=True)
        critic_value = self.critic_model([state_batch, actions], training=True)
        # Used `-value` as we want to maximize the value given
        # by the critic for our actions
        actor_loss = -tf.math.reduce_mean(critic_value)

        return critic_loss, actor_loss

    # We only compute the loss and don't update parameters
    def get_losses(self):
        # get sampling range, if not enough data, batch is small,
        # batch size starting from 1, until reach buffer
        if self.db_server:
            self.logger.info(
                f"start test_pool_sample of size {self.batch_size}.",
                extra=dictLogger,
            )
            batch = self.pool.sample_batch_items(batch_size=self.batch_size)
            assert (
                    len(batch) == self.batch_size
            ), f"sampled batch size {len(batch)} not match sample size {self.batch_size}"

            # convert to tensors
            state = [rec["observation"]["state"] for rec in batch]
            action = [rec["observation"]["action"] for rec in batch]
            reward = [rec["observation"]["reward"] for rec in batch]
            next_state = [rec["observation"]["next_state"] for rec in batch]

            # the shape of the tensor is the same as the buffer
            state_batch = tf.convert_to_tensor(np.array(state))
            action_batch = tf.convert_to_tensor(np.array(action))
            reward_batch = tf.convert_to_tensor(np.array(reward))
            reward_batch = tf.cast(reward_batch, dtype=tf.float32)
            next_state_batch = tf.convert_to_tensor(np.array(next_state))

        else:
            record_range = min(self.buffer_counter, self.buffer_capacity)
            # randomly sample indices , in case batch_size > record_range, numpy default is repeated samples
            batch_indices = np.random.choice(record_range, self.batch_size)

            # convert to tensors
            state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
            action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
            reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
            reward_batch = tf.cast(reward_batch, dtype=tf.float32)
            next_state_batch = tf.convert_to_tensor(
                self.next_state_buffer[batch_indices]
            )

        critic_loss, actor_loss = self.calc_losses(
            state_batch, action_batch, reward_batch, next_state_batch
        )
        return critic_loss, actor_loss

