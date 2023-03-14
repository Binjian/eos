
from dataclasses import dataclass
import numpy as np
import tensorflow as tf
from datetime import datetime

from pymongoarrow.monkey import patch_all

patch_all()

from eos import MongoStore, NPAStore, dictLogger, logger
from eos.config import Truck, trucks_by_name, record_schemas, DB_CONFIG, get_db_config, Record, Episode
from ..dpg import get_algo_data_info

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
    """
    Buffer is the internal dynamic memory object for pooling the experience tuples.
    It can have PoolMixin as storage in mongodb or numpy array file.
    It can provide load(), save(), store(), sample()
    """
    db_key: str = None,  # if None, use npy array for buffer
    truck: Truck = trucks_by_name["VB7"]
    driver: str = "longfei-zheng",
    batch_size: int = 4,
    data_folder: str = "./",
    db_config: DB_CONFIG = None,
    num_states: int = 600,
    num_actions: int = 68,
    buffer_capacity: int = 10000,
    buffer_counter: int = 0,
    query: dict = None,

    def __post_init__(self):

        self.logger = logger.getChild("main").getChild("ddpg").getChild("Buffer")
        self.logger.propagate = True
        # Number of "experiences" to store at max
        # Num of tuples to train on.

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element

        if self.db_key:

            self.db_config = get_db_config(self.db_key)
            self.logger.info(
                f"Using db server {self.db_key} for episode replay buffer..."
            )
            self.db_schema = record_schemas["record_deep"]
            url = self.db_config.Username + ":" + self.db_config.Password + "@" \
                + self.db_config.Host + ":" + self.db_config.Port

            self.query = {"vehicle_id": self.truck.TruckName,
                          "driver_id": self.driver,
                          "dt_start": None,
                          "dt_end": None}

            self.pool = MongoStore(
                location=url,
                collection_type="record",
                mongo_schema=self.db_schema.STRUCTURE,
                query=self.query,
            )


            self.buffer_counter = self.pool.count()
            # check plot with input vehicle and driver
            batch_1 = self.pool.sample(size=1, query=self.query)
            self.num_states, self.num_actions = get_algo_data_info(batch_1[0], self.truck)

            self.logger.info(
                f"Connected to MongoDB {self.db_config.DatabaseName}, "
                f"collection {self.db_config.RecCollName}, "
                f"record number {self.buffer_counter}",
                extra=dictLogger,
            )

        else: #elif self.db_server is '', use the NPAStore as buffer:
            recipe = {'DEFAULT': {'data_folder': self.data_folder,
                                               'recipe_file_name': 'recipe.ini',
                                               'capacity': '300000',
                                               'index': '0',
                                               'full': 'False'},
                                   'array_dims': {'episode_starts': '1',
                                                  'timestamps': '1',
                                                  'states': '600',  # 50*4*3
                                                  'actions': '68',  # 17*4
                                                  'rewards': '1',
                                                  'next_states': '600',  # 50*4*3
                                                  'table_start_rows': '1'}
                      }

            self.pool = NPAStore(location=self.data_folder, recipe=recipe)

    def store_record(self, episode_start_dt, prev_ts, prev_o_t, prev_a_t, prev_table_start, cycle_reward, o_t):
        """
        Store a record in the replay buffer.
        """
        if self.db_key:
            # self.store_record_db(episode_start_dt, prev_ts, prev_o_t, prev_a_t, prev_table_start, cycle_reward, o_t)
            item: Record = {
                "timestamp": datetime.fromtimestamp(
                    float(
                        prev_ts.numpy()[0]
                    )  # fromtimestamp need float, tf data precision set to float32
                ),  # from ms to s
                "plot": {
                    "character": self.truck.TruckName,
                    "driver": self.driver,
                    "when": episode_start_dt,
                    "tz": str(self.truck.tz),
                    "where": "campus",
                    "state_specs": {
                        "observation_specs": [{"velocity_unit": "kmph"},
                                              {"thrust_unit": "percentage"},
                                              {"brake_unit": "percentage"}],
                        "unit_number": self.truck.CloudUnitNumber,  # 4
                        "unit_duration": self.truck.CloudUnitDuration,  # 1s
                        "frequency": self.truck.CloudSignalFrequency,  # 50 hz
                    },  # num_states = length * len(observations) 200*3=600
                    #  length = unit_number * unit_duration  = 4*50=200

                    "action_specs": {
                        "action_row_number": self.truck.ActionFlashRow,
                        "action_column_number": self.truck.PedalScale,
                    },  # num_actions = action_row_number * action_column_number (4*17=68)
                    "reward_specs": {
                        "reward_unit": "wh",
                    },
                },
                "observation": {
                    "state": prev_o_t.numpy().tolist(),
                    "action": prev_a_t.numpy().tolist(),
                    "action_start_row": prev_table_start,
                    "reward": cycle_reward.numpy().tolist(),
                    "next_state": o_t.numpy().tolist(),
                },
            }
        else:
            # self.store_record_npa(episode_start_dt, prev_ts, prev_o_t, prev_a_t, cycle_reward, o_t, prev_table_start)
            item = {
                "episode_starts": episode_start_dt,
                "timestamps": prev_ts,
                "states": prev_o_t,
                "actions": prev_a_t,
                "rewards": cycle_reward,
                "next_states": o_t,
                "table_start_rows": prev_table_start,
            }

        self.pool.store(item=item)
        self.buffer_counter = self.pool.count()  # use query in the initialization of the pool in case of MongoStore
        self.logger.info(f"Pool has {self.buffer_counter} records", extra=dictLogger)

    def sample_minibatch_ddpg(self):
        """
        Update the actor and critic networks using the sampled batch.
        """
        if self.db_key:
            # get sampling range, if not enough data, batch is small
            self.logger.info(
                f"start test_pool_sample of size {self.batch_size, self.truck.TruckName, self.driver}.",
                extra=dictLogger,
            )
            assert self.buffer_counter > 0, "pool is empty"
            batch = self.pool.sample(size=self.batch_size)
            assert (
                    len(batch) == self.batch_size
            ), f"sampled batch size {len(batch)} not match sample size {self.batch_size}"


            # TODO convert output from sample (numpy array) to tensor with tf.convert_to_tensor()

        else:
            batch = self.pool.sample(size=self.batch_size)
            # get sampling range, if not enough data, batch is small,
            # batch size starting from 1, until reach buffer
            # logger.info(f"Tracing!", extra=dictLogger)
            record_range = tf.math.minimum(self.buffer_counter, self.buffer_capacity)
            # randomly sample indices , in case batch_size > record_range, numpy default is repeated samples
            batch_indices = np.random.choice(record_range, self.batch_size)

            # convert to tensors
            state_batch = tf.convert_to_tensor(self.states[batch_indices])
            action_batch = tf.convert_to_tensor(self.actions[batch_indices])
            reward_batch = tf.convert_to_tensor(self.rewards[batch_indices])
            reward_batch = tf.cast(reward_batch, dtype=tf.float32)
            next_state_batch = tf.convert_to_tensor(
                self.next_states[batch_indices]
            )

        return state_batch, action_batch, reward_batch, next_state_batch

