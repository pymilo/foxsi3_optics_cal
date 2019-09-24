"""
---------------------------------------------------
Fitting One 2D Gaussian using LevMarLSQFitter
---------------------------------------------------

Goal: To use astropy.modeling.fitting.LevMarLSQFitter to fit a simple 2D gaussian.
In this section we will get familiarized with the inputs and outputs of the astropy
LevMarLSQFitte function.

Run on terminal: ipython 02_Fit_One_2DGauss.py

Output: Plot Output minus True 2D Gaussians

Date: Jun, 2019
Author: Milo
UC-Berkeley
"""

# Import Packages:
from astropy.modeling import models, fitting
import numpy as np
import matplotlib.pyplot as plt


# Fitting a function requires a *TRUE* set of data and an initial guess to
# then obtained a converged output. Here we mock a "True" 2D-Gaussian with some random noise.

# True 2D-Gaussian
# Parameters for the TRUE 2D-Gaussian:
Tamplitude = 30.
Tx_mean = 100.
Ty_mean = 100.
Tx_stddev = 20.
Ty_stddev = 20.
Ttheta = 0.0
# Define TRUE 2D-Gaussian:
True2dG = models.Gaussian2D(Tamplitude, Tx_mean, Ty_mean, Tx_stddev, Ty_stddev, Ttheta)
# Make X,Y,Z data for the TRUE 2D-Gaussian:
Xg, Yg = np.mgrid[0:201,0:201]
Zg = True2dG(Xg, Yg)

# Initial Guess 2D-Gaussian
# Parameters for the GUESSED 2D-Gaussian:
Gamplitude = 10.
Gx_mean = 60.
Gy_mean = 130.
Gx_stddev = 10.
Gy_stddev = 5.
Gtheta = 5.
# Define GUESSED 2D-Gaussian:
Guess2dG = models.Gaussian2D(10, 60, 130, 10, 5, 5)

# Finding best fit
fit2DG = fitting.LevMarLSQFitter()
Out2dG = fit2DG(Guess2dG,Xg,Yg,Zg)
# Define Z data from the Output 2D-Gaussian function:
Zgo = Out2dG(Xg,Yg)

# Plot Output minus True 2D Gaussians
plt.figure(figsize=(5,5))
ax = plt.axes(projection='3d')
ax.plot_surface(Xg,Yg,np.abs(Zgo-Zg),cmap='viridis')
plt.show()
# Note that the errors between the True and the Output
# 2D-Gaussians are on the order of 1e-15.
