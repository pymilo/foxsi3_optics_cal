'''
---------------------------------------------------
Defining and plotting three 2D-Gaussians
---------------------------------------------------

Goal: To define a function using the
astropy.modeling.models.custom_model decorator to
create the sum of three different 2D-Gaussians.

Run on terminal: ipython 04_Three_2DGauss.py

Output: Plot Output three 2D Gaussians

Date: Jun, 2019
Author: Milo
UC-Berkeley
'''

## Import Packages:
from astropy.modeling import models, fitting
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

''' Definition of Three 2D-Gaussians function '''
@models.custom_model
def ThreeGaussians(x, y,
                 amp1 = 30.,x1_mean = 100.,y1_mean = 100., ## Gauss1 param
                 x1_stddev = 20.,y1_stddev = 20.,theta1 = 0.0,
                 amp2 = 30.,x2_mean = 100.,y2_mean = 100., ## Gauss2 param
                 x2_stddev = 20.,y2_stddev = 20.,theta2 = 0.0,
                 amp3 = 30.,x3_mean = 100.,y3_mean = 100., ## Gauss3 param
                 x3_stddev = 20.,y3_stddev = 20.,theta3 = 0.0,):
    ''' Define Sum of Gauss funtions '''
    g1 = models.Gaussian2D(amp1, x1_mean, y1_mean, x1_stddev, y1_stddev, theta1)
    g2 = models.Gaussian2D(amp2, x2_mean, y2_mean, x2_stddev, y2_stddev, theta2)
    g3 = models.Gaussian2D(amp3, x3_mean, y3_mean, x3_stddev, y3_stddev, theta3)
    return g1(x,y) + g2(x,y) + g3(x,y)

## True parameters:
Tamp1 = 10.;Tx1_mean = 100.;Ty1_mean = 100.;
Tx1_stddev = 10.;Ty1_stddev = 10.;Ttheta1 = 0.0;
Tamp2 = 20.;Tx2_mean = 100.;Ty2_mean = 100.;
Tx2_stddev = 20.;Ty2_stddev = 20.;Ttheta2 = 0.0;
Tamp3 = 30.;Tx3_mean = 100.;Ty3_mean = 100.;
Tx3_stddev = 30.;Ty3_stddev = 30.;Ttheta3 = 0.0;

## Creating a customized ThreeGaussians function :
ThreeGausser = ThreeGaussians(Tamp1,Tx1_mean,Ty1_mean,Tx1_stddev,Ty1_stddev,
                              Ttheta1,Tamp2,Tx2_mean,Ty2_mean,Tx2_stddev,
                              Ty2_stddev,Ttheta2,Tamp3,Tx3_mean,Ty3_mean,
                              Tx3_stddev,Ty3_stddev,Ttheta3)

## Make X,Y,Z data for the THREE 2D-Gaussians:
Xg, Yg = np.mgrid[0:201,0:201]
Zg3 = ThreeGausser(Xg,Yg)

''' Fit Three 2D-Gaussians '''
## Guessed parameters:
Gamp1 = 1.;Gx1_mean = 100.;Gy1_mean = 100.;
Gx1_stddev = 8.;Gy1_stddev = 8.;Gtheta1 = 0.0;
Gamp2 = 2.;Gx2_mean = 110.;Gy2_mean = 110.;
Gx2_stddev = 25.;Gy2_stddev = 25.;Gtheta2 = 0.0;
Gamp3 = 3.;Gx3_mean = 90.;Gy3_mean = 90.;
Gx3_stddev = 35.;Gy3_stddev = 40.;Gtheta3 = 10.0;

## Initial Guess :
ThreeG_guess = ThreeGaussians(Gamp1,Gx1_mean,Gy1_mean,Gx1_stddev,Gy1_stddev,
                              Gtheta1,Gamp2,Gx2_mean,Gy2_mean,Gx2_stddev,
                              Gy2_stddev,Gtheta2,Gamp3,Gx3_mean,Gy3_mean,
                              Gx3_stddev,Gy3_stddev,Gtheta3)
## Finding best fit:
fit2DG = fitting.LevMarLSQFitter()
ThreeG_out = fit2DG(ThreeG_guess,Xg,Yg,Zg3,maxiter=250) ## maxiter keyword is the max number of iterations.


''' Plot Output minus True Three 2D Gaussians '''
Zout = ThreeG_out(Xg,Yg)
plt.figure(figsize=(5,5))
ax = plt.axes(projection='3d')
ax.plot_surface(Xg,Yg,np.abs(Zg3-Zout),cmap='plasma')
plt.show()

'''
From this last example of the three gaussians we confirm that the best is the initial guess of the parameters,
the lower is the number of iterations to make the fit converge. This is a key fact when trying to fit the data
we took at the SLF/MSFC.

'''
