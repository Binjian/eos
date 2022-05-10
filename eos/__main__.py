import sys, os
import pprint

from .agent import *

# from src import PROJROOT
from .comm import *
from .utils import *


# Set the project root directory
# Set the package directory
def main():
    pp = pprint.PrettyPrinter(indent=40)
    pp.pprint(f"veos __main__ CWD: {os.getcwd()}")


if __name__ == "__main__":
    print("test of veos")
    main()
