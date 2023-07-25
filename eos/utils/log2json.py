import argparse
import json
from datetime import datetime

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-p",
    "--path",
    type=str,
    help="log file path, the file to be converted to json format",
)
args = parser.parse_args()

# file name
if args.path:
    # logfile = logfolder + log_file_name
    logfile = args.path
else:
    logfile = "log.log"
# in log file
with open(logfile) as f:
    # line by line
    lines = f.readlines()
    result = []
    for l in lines:
        # remove \n
        l = l.strip("\n")
        # string to dict
        l = json.loads(l)
        created = datetime.fromisoformat(l["asctime"].replace(",", "."))
        l["created"] = created.timestamp()
        # append dict
        result.append(l)

# get log file name
jsonFileName = logfile.split("\.log")[0]
# print(jsonFileName)
# write to json file
jsonFile = open(jsonFileName + ".json", "w")
jsonString = json.dumps(result)
jsonFile.write(jsonString)
jsonFile.close()
