import sys
from dataclasses import dataclass
import argparse

from eos import projroot
from eos.utils import dictLogger, logger

from avatar import Avatar
from algo.ddpg.ddpg import DDPG
from algo.hyperparams import hyper_param_by_name, HYPER_PARAM


@dataclass
class AvatarDDPG(Avatar):
    hyper_param: HYPER_PARAM = hyper_param_by_name('DDPG')

    def __post__init__(self):
        self.agent = DDPG(
            _coll_type='RECORD',
            _hyper_param=self.hyper_param,
            _truck=self.truck,
            _driver=self.driver,
            _pool_key=self.pool_key,
            _data_folder=str(self.data_root),
            _infer_mode=self.infer_mode,
        )

        super().__post_init__()


if __name__ == '__main__':
    """
    ## Setup
    """
    # resumption settings
    parser = argparse.ArgumentParser(
        'Use RL agent (DDPG or RDPG) with tensorflow backend for EOS '
        'with coast-down activated and expected velocity in 3 seconds'
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
        help="User Interface: 'mobile' for mobile phone (for training); 'local' for local hmi; 'cloud' for no UI",
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
        help='relative path to be saved, for create sub folder for different drivers',
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
        '--pool_key',
        type=str,
        default='mongo_local',
        help="url for mongodb server in format usr:password@host:port, "
        "e.g. admint:y02ydhVqDj3QFjT@10.10.0.4:23000, "
        "or simply name with synced default config, e.g. mongo_cluster, mongo_local; "
        "if specified as empty string '', "
        "or \\PATH_TO\\arrow.ini, use arrow pool either in the cluster or local",
    )
    args = parser.parse_args()

    # set up data folder (logging, checkpoint, table)

    assert args.agent == 'ddpg', 'Only DDPG is supported in this module'
    try:
        avatar = AvatarDDPG(
            cloud=args.cloud,
            ui=args.ui,
            resume=args.resume,
            infer_mode=args.infer,
            record=args.record_table,
            path=args.path,
            vehicle_str=args.vehicle,
            driver_str=args.driver,
            remotecan_srv=args.remotecan,
            web_srv=args.web,
            pool_key=args.pool_key,
            proj_root=projroot,
            logger=logger,
        )
    except TypeError as e:
        logger.error(
            f"{{\'header\': \'Project Exception TypeError\', "
            f"\'exception\': \'{e}\'}}",
            extra=dictLogger,
        )
        sys.exit(1)
    except Exception as e:
        logger.error(
            f"{{\'header\': \'main Exception\', " f"\'exception\': \'{e}\'}}",
            extra=dictLogger,
        )
        sys.exit(1)

    avatar.run()
