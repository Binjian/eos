import json

# log filename
logfile = 'test.log'
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
jsonFile = open("data.json","w")
jsonString = json.dumps(result)
jsonFile.write(jsonString)
jsonFile.close()