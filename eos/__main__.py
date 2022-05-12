import os
import pprint
from eos import logger, projroot

from eos.realtime_train_infer_ddpg import realtime_train_infer_ddpg
pp = pprint.PrettyPrinter(indent=40)


# Set the project root directory
# Set the package directory
def main():
    pp.pprint(f"veos __main__ CWD: {os.getcwd()}")
    app = realtime_train_infer_ddpg(
        False, False, True, 'install', projroot, logger
    )
    app.run()


if __name__ == "__main__":
    pp.pprint("test of realtime_train_infer_ddpg")
    main()
