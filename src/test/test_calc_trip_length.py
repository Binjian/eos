import sys, os

sys.path.insert(0, os.path.abspath(".."))

import numpy as np
import pandas as pd
from scipy import integrate

#############################################
# Load Waypoints
#############################################
# Opens the waypoint file and stores it to "waypoints"
excel_data_df = pd.read_excel(
    "../../data/train_data_highring/waypoint_set6.xls", sheet_name="Sheet1"
)
waypoints = excel_data_df.to_numpy()

x = waypoints[:, 0]
y = waypoints[:, 1]
v = waypoints[:, 2]

xdiff = np.diff(x)
ydiff = np.diff(y)
dist_ai = np.sqrt(np.square(xdiff) + np.square(ydiff))

dist_ai_sum = np.sum(dist_ai)

print(dist_ai_sum)
x = np.arange(0, 100)
y = np.square(np.arange(0, 100))

xdiff = np.diff(x)
ydiff = np.diff(y)
dist_ai = np.sqrt(np.square(xdiff) + np.square(ydiff))

dist_ai_sum = np.sum(dist_ai)

print(dist_ai_sum)
