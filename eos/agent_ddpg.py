from dataclasses import dataclass
from agent import Agent
from algo.ddpg.ddpg import DDPG
from algo.hyperparams import hyper_param_by_name, HYPER_PARAM


@dataclass
class AgentDDPG(Agent):
    hyper_param: HYPER_PARAM = hyper_param_by_name('DDPG')

    def __post__init__(self):
        self.algo = DDPG(
            _coll_type='RECORD',
            _hyper_param=self.hyper_param,
            _truck=self.truck,
            _driver=self.driver,
            _pool_key=self.mongo_srv,
            _data_folder=str(self.data_root),
            _infer_mode=self.infer_mode,
        )

        super().__post_init__()
