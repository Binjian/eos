import json

# log filename
logfolder = "../../data/scratch/py_logs/"
logfilename = "l045a_ddpg-episodefree-21-11-29-11-52-35"
logfile = logfolder + logfilename + ".log"
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
# write to json
jsonFile = open("../../data/scratch/json_logs/" + logfilename + ".json", "w")
jsonString = json.dumps(result)
jsonFile.write(jsonString)
jsonFile.close()
