import os
import pprint

from eos import logger, projroot
from eos.RealtimeDDPG import RealtimeDDPG

pp = pprint.PrettyPrinter(indent=40)


# Set the project root directory
# Set the package directory
def main():
    pp.pprint(f"veos __main__ CWD: {os.getcwd()}")
    app = RealtimeDDPG(False, False, True, "install", projroot, logger)
    app.run()


if __name__ == "__main__":
    pp.pprint("test of RealtimeDDPG")
    main()
