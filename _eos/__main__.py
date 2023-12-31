import os
import pprint
import sys

import pandas as pd

from eos import proj_root
from eos.agent import DDPG
from eos.agent.utils import HyperParamDDPG
from eos.avatar import Avatar
from eos.data_io.config import (
    str_to_can_server,
    str_to_driver,
    str_to_trip_server,
    str_to_truck,
)
from eos.data_io.utils import set_root_logger

pp = pprint.PrettyPrinter(indent=40)


# Set the project root directory
# Set the package directory
def main():
    truck = str_to_truck("VB7_FIELD")
    driver = str_to_driver("wang-kai")
    can_server = str_to_can_server("10.0.64.78:5000")
    trip_server = str_to_trip_server("10.0.64.78:9876")
    data_root = proj_root.joinpath("data/" + truck.vin + "-" + driver.pid).joinpath(
        pd.Timestamp.now(truck.site.tz).isoformat()  # has time zone info
    )

    agent = DDPG(
        _coll_type="RECORD",
        _hyper_param=HyperParamDDPG(),
        _truck=truck,
        _driver=driver,
        _pool_key="mongo_local",
        _data_folder=data_root,
        _infer_mode=False,
        _resume=True,
    )
    logger, dict_logger = set_root_logger(
        "eos",
        data_root=data_root,
        agent="ddpg",
        tz=truck.site.tz,
        truck=truck.vid,
        driver=driver.pid,
    )

    try:
        app = Avatar(
            _truck=truck,
            _driver=driver,
            _agent=agent,
            _can_server=can_server,
            _trip_server=trip_server,
            logger=logger,
            dict_logger=dict_logger,
            _resume=True,
            _infer_mode=False,
            data_root=data_root,
        )
    except TypeError as e:
        logger.error(
            f"{{'header': 'Project Exception TypeError', " f"'exception': '{e}'}}",
            extra=dict_logger,
        )
        sys.exit(1)
    except Exception as e:
        logger.error(
            f"{{'header': 'main Exception', " f"'exception': '{e}'}}",
            extra=dict_logger,
        )
        sys.exit(1)
    pp.pprint(f"veos __main__ CWD: {os.getcwd()}")
    app.run()


if __name__ == "__main__":
    pp.pprint("test of RealtimeDDPG")
    main()
