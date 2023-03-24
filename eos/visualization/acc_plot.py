import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# read data from WLTC
df = pd.read_excel('CHTC.xlsx')
v = df['V_mps']
a = df['a_mps2']
a = np.array(a)
# 2D histogram of acceleration
f = plt.figure(1)
n, bins, patches = plt.hist(a, 50, (0, 1.2))
plt.ylabel('Number')
plt.savefig('a_hist_2d.png')
# 2D histogram of velocity
f2 = plt.figure(2)
n, bins, patches = plt.hist(v, 50, (0, 30))
plt.xlabel('Velocity (m/s)')
plt.ylabel('Number')
plt.savefig('v_hist_2d.png')
# 3D histogram of velocity,acceleration
g = plt.figure(3)
ax = g.add_subplot(projection='3d')
hist, xedges, yedges = np.histogram2d(v, a, bins=5, range=[[0, 30], [0, 1.2]])
xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing='ij')
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0
# Construct arrays with the dimensions for the 16 bars.
dx = dy = 0.5 * np.ones_like(zpos)
dz = hist.ravel()
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
ax.set(xlabel='Velocity (m/s)', ylabel='Acceleration (m/s2)', zlabel='Number')
plt.savefig('hist_3d.png')
plt.show()
