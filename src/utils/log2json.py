import json
import argparse
import os
# argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-f",
    "--file",
    type=str,
    help="log file path, the file to be converted to json format",
)
parser.add_argument(
    "-s",
    "--scratch",
    help="log file is in scratc/py_logs or just py_logs",
    action="store_true",
)
args = parser.parse_args()

# log folder
if args.scratch:
    logfolder = "../../data/scratch/py_logs/"
else:
    logfolder = "../../data/py_logs/"

# file name
if args.file:    
    logfilename = args.file
    logfile = logfolder + logfilename
else:
    # default file name
    logfilename = sorted([f for f in os.listdir(logfolder)])[-1]
    print(logfilename)
    logfile = os.path.join(logfolder,logfilename)
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
        # append dict
        result.append(l)

# get log file name
jsonFileName = logfilename.split(".")[0] + "\n"
# write to json file
jsonFile = open("../../data/scratch/json_logs/" + logfilename + ".json", "w")
jsonString = json.dumps(result)
jsonFile.write(jsonString)
jsonFile.close()
