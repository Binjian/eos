import sys
from dataclasses import dataclass
import argparse

from eos import proj_root
from eos.utils import dictLogger, logger
from eos.data_io.config import (
    Driver,
    Truck,
    str_to_truck,
    str_to_driver,
    str_to_can_server,
    str_to_trip_server,
)

from avatar import Avatar  # type: ignore
from agent.ddpg.ddpg import DDPG  # type: ignore
from agent.utils import HyperParamDDPG  # type: ignore


@dataclass
class AvatarDDPG(Avatar):
    hyper_param: HyperParamDDPG = HyperParamDDPG()

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
    try:
        truck: Truck = str_to_truck(args.vehicle)
    except KeyError:
        raise KeyError(f"vehicle {args.vehicle} not found in config file")
    else:
        logger.info(
            f'Vehicle found. vid:{truck.vid}, vin: {truck.vin}.', extra=dictLogger
        )

    try:
        driver: Driver = str_to_driver(args.driver)
    except KeyError:
        raise KeyError(f"driver {args.driver} not found in config file")
    else:
        logger.info(
            f'Driver found. pid:{driver.pid}, vin: {driver.name}.', extra=dictLogger
        )

    # remotecan_srv: str = 'can_intra'
    try:
        can_server = str_to_can_server(args.remotecan)
    except KeyError:
        raise KeyError(f"can server {args.remotecan} not found in config file")
    else:
        logger.info(f'CAN Server found: {can_server.SRVName}', extra=dictLogger)

    try:
        trip_server = str_to_trip_server(args.web)
    except KeyError:
        raise KeyError(f"trip server {args.web} not found in config file")
    else:
        logger.info(f'Trip Server found: {trip_server.SRVName}', extra=dictLogger)
    assert args.agent == 'ddpg', 'Only DDPG is supported in this module'
    agent: DDPG = DDPG(
        _coll_type='RECORD',
        _hyper_param=HyperParamDDPG(),
        _truck=truck,
        _driver=driver,
        _pool_key=args.pool_key,
        _data_folder=args.data_root,
        _infer_mode=args.infer_mode,
    )
    try:
        avatar = AvatarDDPG(
            truck=truck,
            driver=driver,
            can_server=can_server,
            trip_server=trip_server,
            _agent=agent,
            cloud=args.cloud,
            ui=args.ui,
            resume=args.resume,
            infer_mode=args.infer,
            record=args.record_table,
            path=args.path,
            pool_key=args.pool,
            proj_root=proj_root,
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
    finally:
        logger.info(
            f"{{\'header\': \'main finally\', "
            f"\'message\': \'AvatarDDPG created\'}}",
            extra=dictLogger,
        )

    avatar.run()
