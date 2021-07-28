#!/usr/bin/env python
# coding: utf-8

# # create non eco pedal map
# 
# **Author:** binjian Xin<br>
# **Date Created:** 2021/07/07<br>
# **Last Modified:** 2021/07/07<br>
# **Description:** create non-eco pedal map and visualize<br>
# 
# ## Create the non-eco pedal map
# 
# read default pedal map (claiming to be eco) from data file

# In[1]:


import numpy as np
import pandas as pd

pd_data0 = pd.read_csv("../../data/init_table")
# create a matplotlib 3d figure, //export and save in log
# pd_data0.columns = np.linspace(0, 1.0, num=17)
# pd_data0.index = np.linspace(0, 30, num=21)
# vcu_calib_table_0 = pd_data0.to_numpy()
#
#


# 

# In[2]:




# pd_data0 = pd.DataFrame(
#     vcu_calib_table_0,
#     columns=np.linspace(0, 1.0, num=17),
#     index=np.linspace(0, 30, num=21),
# )


# ## Visualization
# ### plotly
# show default pedal map

# ### Preliminaries for plotly and chart studio

# In[3]:


import plotly.graph_objects as go

sh_0, sh_1 = pd_data0.shape
x, y =  np.linspace(0,1.0,sh_1), np.linspace(0,30, sh_0)
z = pd_data0.values
# # Download data set from plotly repo
# pts = np.loadtxt(np.DataSource().open('https://raw.githubusercontent.com/plotly/datasets/master/mesh_dataset.txt'))
# x, y, z = pts.T
# fig1 = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, color='lightpink', opacity=0.50)])
# fig1.show()

z0 = np.zeros(z.shape)
figure0 = go.Figure(data=[
                        go.Surface(
                        contours = {
                            "y": {"show": True, "start":0, "end":10, "size":0.5, "color":"cyan"},
                            "z": {"show": True, "start":-3000, "end":4600, "size":100, "color":"blue"}
                        },
                        x=x,
                        y=y,
                        z=pd_data0.values),
                        go.Surface(x=x,y=y,z=z0)
                        ]
                  )
# figure.update_traces(contours_z=dict(show=True, usecolormap=True,
#                                   highlightcolor="limegreen", project_z=True))
figure0.update_layout(title='Pedal Map', autosize=False,
                     scene=dict(
                         xaxis_title='pedal',
                         yaxis_title='velocity',
                         zaxis_title='torque'),
                     width=700, height=700,
                     margin=dict(l=65,r=50,b=65,t=90))
# figure0.add_trace(go.Surface(x=x,y=y,z=z0, surfacecolor=np.ones(z0.shape)))
figure0.show()

