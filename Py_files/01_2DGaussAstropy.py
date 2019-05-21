'''
---------------------------------------------------
Script that plot a 2D Gaussian using astropy function
---------------------------------------------------

Run on terminal: ipython 01_2DGaussAstropy.py

Output: 2D Gauss + radom noise plot on a 3D projection

Date: May, 2019
Author: Milo
UC-Berkeley
'''
## Import packages
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from astropy.modeling import powerlaws, fitting, models

## Input parameters:
amplitude = 30.
x_mean = 100.
y_mean = 100.
x_stddev = 20.
y_stddev = 20.
theta = 0.0
## 2D Gaussian using initial parameters:
g2di = models.Gaussian2D(amplitude, x_mean, y_mean, x_stddev, y_stddev, theta)
#print(g2di)
Xg, Yg = np.mgrid[0:201,0:201]
Zg = g2di(Xg, Yg)

plt.figure(figsize=(6,6))
ax = plt.axes(projection='3d')
ax.plot_surface(Xg,Yg,Zg,cmap='viridis')
plt.show()







