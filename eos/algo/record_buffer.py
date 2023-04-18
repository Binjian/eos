from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import tensorflow as tf
from datetime import datetime
from typing import Optional

from pymongoarrow.monkey import patch_all


from eos import DBPool, RecordFilePool, dictLogger, logger
from eos.config import (
    Truck,
    trucks_by_name,
    DB_CONFIG,
    get_db_config,
)
from eos.struct import RecordDoc, RecordArr, Plot, ObservationSpecs

from .dpg import get_algo_data_info

patch_all()

"""
We use [OpenAIGym](https://gymnasium.farama.org/) to create the environment.
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

    db_key: Optional[str] = None  # if None, use npy array for buffer
    truck: Truck = trucks_by_name['VB7']
    driver: str = ('longfei-zheng',)
    batch_size: int = (4,)
    padding_value: float = (0,)
    data_folder: str = ('./',)
    db_config: DB_CONFIG = (None,)
    num_states: int = (600,)
    num_actions: int = (68,)
    buffer_capacity: int = (10000,)
    buffer_counter: int = (0,)
    query: dict = (None,)

    def __post_init__(self):
        self.logger = logger.getChild('main').getChild('ddpg').getChild('Buffer')
        self.logger.propagate = True
        # Number of "experiences" to store at max
        # Num of tuples to train on.

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element

        # TODO use type() and match to select the right buffer handling
        if self.db_key:
            self.db_config = get_db_config(self.db_key)
            self.logger.info(
                f'Using db server {self.db_key} for episode replay buffer...'
            )
            url = (
                self.db_config.Username
                + ':'
                + self.db_config.Password
                + '@'
                + self.db_config.Host
                + ':'
                + self.db_config.Port
            )

            self.query = {
                'vehicle_id': self.truck.TruckName,
                'driver_id': self.driver,
                'dt_start': None,
                'dt_end': None,
            }

            self.pool = DBPool[RecordDoc](
                location=url,
                query=self.query,
            )

            self.buffer_counter = self.pool.count()
            # check plot with input vehicle and driver
            batch_1 = self.pool.sample(size=1, query=self.query)
            self.num_states, self.num_actions = get_algo_data_info(
                batch_1[0], self.truck
            )

            self.logger.info(
                f'Connected to MongoDB {self.db_config.DatabaseName}, '
                f'collection {self.db_config.RecCollName}, '
                f'record number {self.buffer_counter}',
                extra=dictLogger,
            )

        else:  # elif self.db_key is None, use the NPAStore as buffer:
            recipe = {
                'DEFAULT': {
                    'data_folder': self.data_folder,
                    'recipe_file_name': 'recipe.ini',
                    'capacity': '300000',
                    'index': '0',
                    'full': 'False',
                },
                'array_dims': {
                    'episode_starts': '1',
                    'timestamps': '1',
                    'states': '600',  # 50*4*3
                    'actions': '68',  # 17*4
                    'rewards': '1',
                    'next_states': '600',  # 50*4*3
                    'table_start_rows': '1',
                },
            }

            self.pool = RecordFilePool(location=self.data_folder, recipe=recipe)

    def store_record(
        self,
        episode_start_dt,
        prev_ts,
        prev_o_t,
        prev_a_t,
        prev_table_start,
        cycle_reward,
        o_t,
    ):
        """
        Store a record in the replay buffer.
        """
        plot = Plot(
            character=self.truck.TruckName,
            driver=self.driver,
            when=self.episode_start_dt,
            tz=str(self.truck.tz),
            where=self.truck.Location,
            state_specs={
                'observation_specs': ObservationSpecs(
                    velocity_unit='kph',
                    thrust_unit='pct',
                    brake_unit='pct',
                ),
                'unit_number': self.truck.CloudUnitNumber,  # 4
                'unit_duration': self.truck.CloudUnitDuration,  # 1s
                'frequency': self.truck.CloudSignalFrequency,  # 50 hz
            },
            action_specs={
                'action_row_number': self.truck.ActionFlashRow,
                'action_column_number': self.truck.PedalScale,
            },
            reward_specs={
                'reward_unit': 'wh',
            },
        )
        if self.db_key:
            # self.store_record_db(episode_start_dt, prev_ts, prev_o_t, prev_a_t, prev_table_start, cycle_reward, o_t)

            item: RecordDoc = {
                'timestamp': datetime.fromtimestamp(
                    float(
                        prev_ts.numpy()[0]
                    )  # fromtimestamp need float, tf data precision set to float32
                ),  # from ms to s
                'plot': plot,
                'observation': {
                    'timestamp': prev_ts,
                    'state': prev_o_t.numpy().tolist(),
                    'action': prev_a_t.numpy().tolist(),
                    'action_start_row': prev_table_start,
                    'reward': cycle_reward.numpy().tolist(),
                    'next_state': o_t.numpy().tolist(),
                },
            }
        else:
            # self.store_record_npa(episode_start_dt, prev_ts, prev_o_t, prev_a_t, cycle_reward, o_t, prev_table_start)
            item: RecordArr = {
                'episode_starts': episode_start_dt,
                'plot': Plot,
                'timestamps': prev_ts.numpy(),
                'states': prev_o_t.numpy(),
                'actions': prev_a_t.numpy(),
                'rewards': cycle_reward,
                'next_states': o_t.numpy(),
                'table_start_rows': prev_table_start,
            }

        self.pool.store(item=item)
        self.buffer_counter = (
            self.pool.count()
        )  # use query in the initialization of the pool in case of MongoStore
        self.logger.info(f'Pool has {self.buffer_counter} records', extra=dictLogger)

    def sample_minibatch_record(self):
        """
        Update the actor and critic networks using the sampled batch.
        """
        assert self.buffer_counter > 0, 'pool is empty'
        # get sampling range, if not enough data, batch is small
        self.logger.info(
            f'start sample from pool with size: {self.batch_size}, '
            f'truck: {self.truck.TruckName}, driver: {self.driver}.',
            extra=dictLogger,
        )

        batch = self.pool.sample(size=self.batch_size)
        assert (
            len(batch) == self.batch_size
        ), f'sampled batch size {len(batch)} not match sample size {self.batch_size}'

        states = [rec['observation']['state'] for rec in batch]
        actions = [rec['observation']['action'] for rec in batch]
        rewards = [rec['observation']['reward'] for rec in batch]
        next_states = [rec['observation']['next_state'] for rec in batch]

        # convert output from sample (list or numpy array) to tf.tensor
        states = tf.convert_to_tensor(np.array(states), dtype=tf.float32)
        actions = tf.convert_to_tensor(np.array(actions), dtype=tf.float32)
        rewards = tf.convert_to_tensor(np.array(rewards), dtype=tf.float32)
        next_states = tf.convert_to_tensor(np.array(next_states), dtype=tf.float32)

        return states, actions, rewards, next_states
