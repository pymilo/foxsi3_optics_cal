'''
---------------------------------------------------
One 2D Gaussian by using astropy.modeling.models.Gaussian2D
---------------------------------------------------

Goal: To use astropy to plot a single 2D Gaussian using Mayavi.
Mayavi provides a smoother way to display and rotate the gaussians. 
It opens on a separe window.

Run on terminal: ipython 01_2DGauss_Astropy.py

Output: Mayavi Plot 2D Gauss

Date: Jun, 2019
Author: Milo
UC-Berkeley
'''

## Import Packages:
import numpy as np
from astropy.modeling import models
from mayavi import mlab

## Input parameters:
amplitude = 30.
x_mean = 100.
y_mean = 100.
x_stddev = 20.
y_stddev = 20.
theta = 0.0

## 2D Gaussian using initial parameters:
g2di = models.Gaussian2D(amplitude, x_mean, y_mean, x_stddev, y_stddev, theta)
Xg, Yg = np.mgrid[0:201,0:201]
Zg = g2di(Xg, Yg)

## Plot 2D Gaussian using Mayavi
mlab.surf(Zg,warp_scale='auto')
mlab.axes()
mlab.show()




