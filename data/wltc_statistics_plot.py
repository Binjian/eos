import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read data from WLTC
df = pd.read_excel("./WLTC.xlsx")
v = df["velocity"] / 3.6
# calculate acceleration
a = np.diff(v)
a = np.insert(a, 0, 0)
# 2D histogram of acceleration
f = plt.figure(1)
n, bins, patches = plt.hist(a, 100, (-1.7, 1.7))
plt.xlabel("Acceleration (m/s2)")
plt.ylabel("Number")
plt.savefig("./hist_2d_acc.png")

# 3D histogram of velocity,acceleration
g = plt.figure(2)
ax = g.add_subplot(projection="3d")
hist, xedges, yedges = np.histogram2d(v, a, bins=20, range=[[0, 40], [-1.7, 1.7]])
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0
# Construct arrays with the dimensions for the 16 bars.
dx = dy = 0.5 * np.ones_like(zpos)
dz = hist.ravel()
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort="average", cmap='plasma')
ax.set(xlabel="Velocity (m/s)", ylabel="Acceleration (m/s2)", zlabel="Number")
plt.savefig("./hist_3d.png")

# 2D histogram of speed
h = plt.figure(3)
n1, bins1, patches1 = plt.hist(v, 100, (0, 40))
plt.xlabel("Speed (m/s)")
plt.ylabel("Number")
plt.savefig("./hist_2d_spd.png")

plt.show()
