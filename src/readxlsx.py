import pandas as pd
import numpy as np
import time


df = pd.read_excel("WLTC.xlsx", sheet_name="WLTC")

v_real = np.array(df["velocity"])

counter = 0

time.sleep(2)

while counter < 1800:
    print("time:", str(counter), "s")
    print(v_real[counter])
    counter = counter + 1
    time.sleep(1.13)


print("finish simulation!")
