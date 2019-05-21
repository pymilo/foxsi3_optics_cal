'''
---------------------------------------------------
Simple function that defines and plot a 2D gaussian
---------------------------------------------------

Run on terminal: ipython 00_2DGauss_Func.py

Output: Plot 2D Gauss + radom noise plot on a 3D projection

Date: May, 2019
Author: Milo
UC-Berkeley
'''

## Import needed Packages:
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

## Function definition:
def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

## Create the gaussian data
Xin, Yin = np.mgrid[0:201, 0:201]
gdata = gaussian(30, 100, 100, 20, 20)(Xin, Yin) + np.random.random(Xin.shape)

## Plot:
fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
ax.plot_surface(Xin,Yin,gdata,cmap='viridis')
plt.show()

