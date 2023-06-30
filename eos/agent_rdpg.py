from dataclasses import dataclass
import argparse, sys
from agent import Agent
from algo.rdpg import RDPG
from algo.hyperparams import hyper_param_by_name, HYPER_PARAM
from eos.utils import dictLogger, logger
from eos import projroot


@dataclass
class AgentRDPG(Agent):
    hyper_param: HYPER_PARAM = hyper_param_by_name('RDPG')

    def __post_init__(self):
        self.algo = RDPG(
            _coll_type='EPISODE',
            _truck=self.truck,
            _driver=self.driver,
            _pool_key=self.mongo_srv,
            _data_folder=str(self.data_root),
            _hyper_param=self.hyper_param,
            _infer_mode=self.infer_mode,
        )
        super().__post_init__()


if __name__ == '__main__':
    """
    ## Setup
    """
    # resumption settings
    parser = argparse.ArgumentParser(
        'Use RL agent (DDPG or RDPG) with tensorflow backend for EOS with coastdown activated and expected velocity in 3 seconds'
    )
    parser.add_argument(
        '-a',
        '--agent',
        type=str,
        default='ddpg',
        help="RL agent choice: 'ddpg' for DDPG; 'rdpg' for Recurrent DPG",
    )

    parser.add_argument(
        '-c',
        '--cloud',
        default=False,
        help='Use cloud mode, default is False',
        action='store_true',
    )

    parser.add_argument(
        '-u',
        '--ui',
        type=str,
        default='cloud',
        help="User Inferface: 'mobile' for mobile phone (for training); 'local' for local hmi; 'cloud' for no UI",
    )

    parser.add_argument(
        '-r',
        '--resume',
        default=True,
        help='resume the last training with restored model, checkpoint and pedal map',
        action='store_true',
    )

    parser.add_argument(
        '-i',
        '--infer',
        default=False,
        help='No model update and training. Only Inference mode',
        action='store_true',
    )
    parser.add_argument(
        '-t',
        '--record_table',
        default=True,
        help='record action table during training',
        action='store_true',
    )
    parser.add_argument(
        '-p',
        '--path',
        type=str,
        default='.',
        help='relative path to be saved, for create subfolder for different drivers',
    )
    parser.add_argument(
        '-v',
        '--vehicle',
        type=str,
        default='.',
        help="vehicle ID like 'VB7' or 'MP3' or VIN 'HMZABAAH1MF011055'",
    )
    parser.add_argument(
        '-d',
        '--driver',
        type=str,
        default='.',
        help="driver ID like 'longfei.zheng' or 'jiangbo.wei'",
    )
    parser.add_argument(
        '-m',
        '--remotecan',
        type=str,
        default='10.0.64.78:5000',
        help='url for remote can server, e.g. 10.10.0.6:30865, or name, e.g. baiduyun_k8s, newrizon_test',
    )
    parser.add_argument(
        '-w',
        '--web',
        type=str,
        default='10.0.64.78:9876',
        help='url for web ui server, e.g. 10.10.0.13:9876, or name, e.g. baiduyun_k8s, newrizon_test',
    )
    parser.add_argument(
        '-o',
        '--mongodb',
        type=str,
        default='mongo_local',
        help="url for mongodb server in format usr:password@host:port, e.g. admint:y02ydhVqDj3QFjT@10.10.0.4:23000, or simply name with synced default config, e.g. mongo_cluster, mongo_local; if specified as empty string '', use local npy file",
    )
    args = parser.parse_args()

    # set up data folder (logging, checkpoint, table)

    try:
        app = AgentRDPG(
            cloud=args.cloud,
            ui=args.ui,
            resume=args.resume,
            infer=args.infer,
            record=args.record_table,
            path=args.path,
            vehicle=args.vehicle,
            driver=args.driver,
            remotecan_srv=args.remotecan,
            web_srv=args.web,
            mongo_srv=args.mongodb,
            proj_root=projroot,
            logger=logger,
        )
    except TypeError as e:
        logger.error(f'Project Exeception TypeError: {e}', extra=dictLogger)
        sys.exit(1)
    except Exception as e:
        logger.error(e, extra=dictLogger)
        sys.exit(1)
    app.run()
