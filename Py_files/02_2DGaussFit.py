'''
---------------------------------------------------
Script that fits a 2D Gaussian using LevMarLSQFitter
---------------------------------------------------

Run on terminal: ipython 02_2DGaussFit.py

Output: Fitting parameters of a 2D Gaussian
        Print the real, guessed, and fited gaussian paramenters.
        

Date: May, 2019
Author: Milo
UC-Berkeley
'''
## Import packages
import numpy as np
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
print('Real Gaussian Parameters:')
print(g2di)

## Create grids for X, Y and Z:
Xg, Yg = np.mgrid[0:201,0:201]
Zg = g2di(Xg, Yg)

## 2DGauss guess:
g2d_guess = models.Gaussian2D(10, 60, 130, 10, 5, 5)
print('Guessed Gaussian Parameters:')
print(g2d_guess)


## Finding best fit:
fit2DG = fitting.LevMarLSQFitter()
g2d_out = fit2DG(g2d_guess,Xg,Yg,Zg)
print('Fitted Gaussian Parameters:')
print(g2d_out)








