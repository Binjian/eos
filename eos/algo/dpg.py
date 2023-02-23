import abc
from dataclasses import dataclass
from datetime import datetime


from eos import Pool, dictLogger, logger
from eos.config import (
    db_servers_by_name,
    db_servers_by_host,
    episode_schemas,
    Truck,
    trucks_by_name,
)

"""Base class for differentiable policy gradient methods."""


@dataclass
class DPG(abc.ABC):
    _truck: Truck = (trucks_by_name["VB7"],)
    _driver: str = ("longfei",)
    _num_states: int = (600,)
    _num_actions: int = (68,)
    _buffer_capacity: int = (10000,)
    _batch_size: int = (4,)
    _hidden_unitsAC: tuple = ((256, 16, 32),)
    _action_bias: float = (0.0,)
    _n_layersAC: tuple = ((2, 2),)
    _padding_value: float = (0,)
    _gamma: float = (0.99,)
    _tauAC: tuple = ((0.005, 0.005),)
    _lrAC: tuple = ((0.001, 0.002),)
    _datafolder: str = ("./",)
    _ckpt_interval: int = (5,)
    _db_server: str = ("mongo_local",)
    _resume: bool = (True,)
    _infer_mode: bool = (False,)
    _pool: Pool = (None,)

    def __post_init__(self):
        self.logger = logger.getchild("main").getchild(self.__str__())
        self.logger.propagate = True
        self.dictLogger = dictLogger

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
                f"Using db server {self.db_server} for episode replay buffer..."
            )

        # Number of "experiences" to store     at max
        # Num of tuples to train on.
        self.touch_gpu()

    def __del__(self):
        if self.db_server:
            # for database, exit needs drop interface.
            self.pool.drop_mongo()
        else:
            self.save_replay_buffer()

    def __repr__(self):
        return f"DPG({self.truck.name}, {self.driver})"

    def __str__(self):
        return "DPG"

    @abc.abstractmethod
    def touch_gpu(self):
        pass

    @abc.abstractmethod
    def init_checkpoint(self):
        # Actor create or restore from checkpoint
        # add checkpoints manager
        pass

    @abc.abstractmethod
    def actor_predict(self, obs, t):
        """
        Evaluate the actors given a single observations.
        Batchsize is 1.
        """
        pass

    def start_episode(self, dt: datetime):
        self.logger.info(f"Episode start at {dt}", extra=dictLogger)
        # somehow mongodb does not like microseconds in rec['plot']
        dt_milliseconds = int(dt.microsecond / 1000) * 1000
        self.episode_start_dt = dt.replace(microsecond=dt_milliseconds)
        self.h_t = []

    @abc.abstractmethod
    def deposit(self, prev_ts, prev_o_t, prev_a_t, prev_table_start, cycle_reward, o_t):
        """Deposit the experience into the replay buffer."""
        pass

    @abc.abstractmethod
    def end_episode(self):
        """Deposit the whole episode of experience into the replay buffer for RDPG.
        Noting to do for DDPG"""
        pass

    @abc.abstractmethod
    def train(self):
        """
        Train the actor and critic moving network.

        return:
            tuple: (actor_loss, critic_loss)
        """
        pass

    @abs.abstractmethod
    def get_losses(self):
        """
        Get the actor and critic losses without calculating the gradients.
        """
        pass

    @abc.abstractmethod
    def soft_update_target(self):
        """
        update target networks with tiny tau value, typical value 0.001.
        done after each batch, slowly update target by Polyak averaging.
        """
        pass

    @abc.abstractmethod
    def save_ckpt(self):
        """
        save checkpoints of actor and critic
        """
        pass

    @abc.abstractmethod
    def save_replay_buffer(self):
        """
        save replay buffer when exit algo
        """
        pass

    @abc.abstractmethod
    def load_replay_buffer(self):
        """
        load replay buffer when start algo
        """
        pass

    @property
    def db_server(self):
        return self._db_server

    @db_server.setter
    def db_server(self, value):
        raise AttributeError("db_server is read-only")

    @property
    def truck(self):
        return self._truck

    @truck.setter
    def truck(self, value):
        raise AttributeError("truck is read-only")

    @property
    def driver(self):
        return self._driver

    @driver.setter
    def driver(self, value):
        raise AttributeError("driver is read-only")

    @property
    def num_states(self):
        return self._num_states

    @num_states.setter
    def num_states(self, value):
        raise AttributeError("num_states is read-only")

    @property
    def num_actions(self):
        return self._num_actions

    @num_actions.setter
    def num_actions(self, value):
        raise AttributeError("num_actions is read-only")

    @property
    def buffer_capacity(self):
        return self._buffer_capacity

    @buffer_capacity.setter
    def buffer_capacity(self, value):
        raise AttributeError("seq_len is read-only")

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        raise AttributeError("batch_size is read-only")

    @property
    def hidden_unitsAC(self):
        return self._hidden_unitsAC

    @hidden_unitsAC.setter
    def hidden_unitsAC(self, value):
        raise AttributeError("hidden_unitsAC is read-only")

    @property
    def action_bias(self):
        return self._action_bias

    @action_bias.setter
    def action_bias(self, value):
        raise AttributeError("action_bias is read-only")

    @property
    def n_layersAC(self):
        return self._n_layersAC

    @n_layersAC.setter
    def n_layersAC(self, value):
        raise AttributeError("n_layersAC is read-only")

    @property
    def padding_value(self):
        return self._padding_value

    @padding_value.setter
    def padding_value(self, value):
        raise AttributeError("padding_value is read-only")

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        raise AttributeError("gamma is read-only")

    @property
    def tauAC(self):
        return self._tauAC

    @tauAC.setter
    def tauAC(self, value):
        raise AttributeError("tauAC is read-only")

    @property
    def lrAC(self):
        return self._lrAC

    @lrAC.setter
    def lrAC(self, value):
        raise AttributeError("lrAC is read-only")

    @property
    def datafolder(self):
        return self._datafolder

    @datafolder.setter
    def datafolder(self, value):
        raise AttributeError("datafolder is read-only")

    @property
    def ckpt_interval(self):
        return self._ckpt_interval

    @ckpt_interval.setter
    def ckpt_interval(self, value):
        raise AttributeError("ckpt_interval is read-only")

    @property
    def db_server(self):
        return self._db_server

    @db_server.setter
    def db_server(self, value):
        raise AttributeError("db_server is read-only")

    @property
    def resume(self):
        return self._resume

    @resume.setter
    def resume(self, value):
        raise AttributeError("resume is read-only")

    @property
    def infer_mode(self):
        return self._infer_mode

    @infer_mode.setter
    def infer_mode(self, value):
        raise AttributeError("infer_mode is read-only")

    @property
    def pool(self):
        return self._pool

    @pool.setter
    def pool(self, value):
        raise AttributeError("pool is read-only")
