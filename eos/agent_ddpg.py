from dataclasses import dataclass
from agent import Agent
from algo.ddpg.ddpg import DDPG


@dataclass
class AgentDDPG(Agent):
    # Learning rate for actor-critic models
    critic_lr: float = (0.002,)
    actor_lr: float = 0.001
    # Discount factor for future rewards
    gamma: float = 0.99
    # Used to update target networks
    tau_ac: tuple = (0.005, 0.005)
    hidden_units_ac: tuple = (256, 16, 32)
    action_bias: float = 0
    lr_ac: tuple = (0.001, 0.002)
    seq_len: int = 8  # TODO  7 maximum sequence length
    buffer_capacity: int = 300000
    batch_size: int = 4
    # number of hidden units in the actor and critic networks
    # number of layer in the actor-critic network
    n_layers_ac: tuple = (2, 2)
    # padding value for the input, impossible value for observation, action or reward
    padding_value: int = -10000
    ckpt_interval: int = 5

    def __post__init__(self):
        self.algo = DDPG(
            _truck=self.truck,
            _driver=self.driver,
            _num_states=self.num_states,
            _num_actions=self.num_actions,
            _buffer_capacity=self.buffer_capacity,
            _batch_size=self.batch_size,
            _hidden_units_ac=self.hidden_units_ac,
            _action_bias=self.action_bias,
            _n_layers_ac=self.n_layers_ac,
            _padding_value=self.padding_value,
            _gamma=self.gamma,
            _tau_ac=self.tau_ac,
            _lr_ac=self.lr_ac,
            _data_folder=str(self.data_root),
            _ckpt_interval=self.ckpt_interval,
            _infer_mode=self.infer_mode,
        )

        super().__post_init__()
