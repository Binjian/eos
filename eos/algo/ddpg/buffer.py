from dataclasses import dataclass
import numpy as np
import tensorflow as tf

from pymongoarrow.monkey import patch_all

patch_all()

from eos import Pool, dictLogger, logger
from eos.config import DB, record_schemas, Truck, trucks_by_name

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
class Buffer:
    db: DB = None,  # if None, use npy array for buffer
    truck: Truck = trucks_by_name["VB7"]
    driver: str = "longfei",
    num_states: int = 600,
    num_actions: int = 68,
    batch_size: int = 4,
    buffer_capacity: int = 10000,
    datafolder: str = "./",
    file_sb: str = None
    file_ab: str = None
    file_rb: str = None
    file_nsb: str = None
    file_bc: str = None
    state_buffer: np.ndarray = None
    action_buffer: np.ndarray = None
    reward_buffer: np.ndarray = None
    next_state_buffer: np.ndarray = None

    def __post_init__(
        self,
    ):
        self.logger = logger.getChild("main").getChild("ddpg").getChild("Buffer")
        self.logger.propagate = True
        # Number of "experiences" to store at max
        # Num of tuples to train on.

        if self.db:
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
            batch_4 = self.pool.sample_batch_items(
                batch_size=4, vehicle_id=self.truck.TruckName
            )
            obs = batch_4[0]["plot"]["states"]["observations"]
            unit_number = batch_4[0]["plot"]["states"]["unit_number"]
            unit_duration = batch_4[0]["plot"]["states"]["unit_duration"]
            frequency = batch_4[0]["plot"]["states"]["frequency"]
            self.num_states = len(obs) * unit_number * unit_duration * frequency

            action_row_number = batch_4[0]["plot"]["actions"]["action_row_number"]
            action_column_number = batch_4[0]["plot"]["actions"]["action_column_number"]
            self.num_actions = action_row_number * action_column_number

            self.logger.info(
                f"Connected to MongoDB {self.db.DatabaseName}, collection {self.db.RecCollName}, record number {self.buffer_counter}",
                extra=dictLogger,
            )
        else:  # elif self.db is '':
            self.buffer_capacity = tf.convert_to_tensor(
                self.buffer_capacity, dtype=tf.int64
            )
            self.file_sb = self.datafolder + "/state_buffer.npy"
            self.file_ab = self.datafolder + "/action_buffer.npy"
            self.file_rb = self.datafolder + "/reward_buffer.npy"
            self.file_nsb = self.datafolder + "/next_state_buffer.npy"
            self.file_bc = self.datafolder + "/buffer_counter.npy"
            # Its tells us num of times record() was called.
            self.buffer_counter = tf.convert_to_tensor(0, dtype=tf.int64)
            self.load_replay_buffer()

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element

    def save_replay_buffer(self):
        np.save(self.file_sb, self.state_buffer)
        np.save(self.file_ab, self.action_buffer)
        np.save(self.file_rb, self.reward_buffer)
        np.save(self.file_nsb, self.next_state_buffer)
        np.save(self.file_bc, self.buffer_counter)
        self.logger.info(f"saved buffer counter: {self.buffer_counter}")

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
            self.state_buffer = np.zeros((self.buffer_capacity, self.num_states))
            self.action_buffer = np.zeros((self.buffer_capacity, self.num_actions))
            self.reward_buffer = np.zeros((self.buffer_capacity, 1))
            self.next_state_buffer = np.zeros((self.buffer_capacity, self.num_states))
            self.buffer_counter = 0
            print("blank experience")

    def load_replay_buffer(self):
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

    def store_db(self, rec: dict):
        """
        Record a new experience in the pool (database).
        """
        result = self.pool.deposit_item(rec)
        assert result.acknowledged == True, "Record not deposited!"
        rec_inserted = self.pool.find_item(result.inserted_id)
        assert (
            rec_inserted == rec
        ), "Record inserted is not the same as the one inserted!"
        self.buffer_counter = self.pool.count_items(
            vehicle_id=self.truck.TruckName, driver_id=self.driver
        )
        self.logger.info(f"Pool has {self.buffer_counter} records", extra=dictLogger)

    # Takes (s,a,r,s') obervation tuple as input
    def store_npa(self, obs_tuple: tuple):
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


    # We compute the loss and update parameters
    def sample_minibatch_ddpg(self):
        """
        Update the actor and critic networks using the sampled batch.
        """
        if self.db:
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

        return state_batch, action_batch, reward_batch, next_state_batch
