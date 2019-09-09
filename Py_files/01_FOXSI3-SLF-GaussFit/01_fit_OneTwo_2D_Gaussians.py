'''
---------------------------------------------------
Simple way to fit One and Two 2D-Gaussians for FOXSI3-SLF Data (Dark corrected)
---------------------------------------------------

Goal: To fit One and Two 2D-Gaussians to the FOXSI3 SLF data corrected by darks.

Input:  1. Fits file with SLF Data taken with the Andor CCD Camera.
        2. Dark fits files for correction.

Run on terminal: ipython 01_fit_OneTwo_2D_Gaussians.py

Output:
            1. Flat plots of the One 2D-Gaussians in Log scale.
            2. 3D plots of the One 2D-Gaussians.
            3. Plot of the difference Data vs. Fit in log scale.
            4. Print out the parameters for the three 2D-Gaussians and the offset.
            5. Flat plots of the Two 2D-Gaussians in Log scale for One 2D-Gaussian.
            6. 3D plots of the Two 2D-Gaussians.
            7. Plot of the difference Data vs. Fit in log scale for Two 2D-Gaussians.
            8. Print out the parameters for the Two 2D-Gaussians and the offset.

Date: Sep, 2019
Author: Milo
UC-Berkeley
'''
## Import Packages:
from astropy.io import fits as pyfits
from astropy import wcs
from ndcube import NDCube
from astropy.modeling import models, fitting
from astropy.visualization import ImageNormalize, MinMaxInterval, LogStretch#, LinearStretch
from mpl_toolkits import mplot3d
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.stats

## Workspace:
''' R^2 function definition '''
def RSquared(data,model):
    ## The closest to 1, the better is the fit.
    ss_err=(model.fit_info['fvec']**2).sum()
    ss_tot=((data-data.mean())**2).sum()
    return 1-(ss_err/ss_tot)

## Read data
''' Plate scale '''
distance = 2. * u.m ## FOXSI focal distance
CCD_PIXEL_PITCH = 13.5 * u.micron ## Andor CCD Pitch in microns
binning = 2. ## binning used for collecting data.
plate_scale = np.arctan(binning * CCD_PIXEL_PITCH / distance).to(u.arcsec) ## in arcsec

''' Path to the folder where to find your data and darks: '''
folder = '/Users/Kamilobu/Desktop/Developer/foxsi3_optics_cal/data/'

''' File names: '''
filename = 'kFOXSI3_X10-Test_CCD_T2Sx6_10kV_0p02mA_0mmZ.fits'  ## name of your data fits file.
darkfilename = 'Dark1_FOXSI3_X10-Test_CCD_T2Sx6_10kV_0p02mA_+15mmZ.fits'  ## name of your darks fits file.
## These are fits files containing six frames each of 1024x1024 pixels taken at the SLF
## using the Andor camera and the Mo X-ray source. Voltages,Currents and Integration Times are
## indicated over the names of the files.

## Read fits files using astropy.io.fits
fits = pyfits.open(folder+filename)
darkfits = pyfits.open(folder+darkfilename)

## Create data array corrected by darks:
data = np.average(fits[0].data, axis=0) - np.average(darkfits[0].data, axis=0)
smax_pixel = np.unravel_index(np.argmax(data), data.shape)
fov = [20, 20] ## [px,px]
sdata = data[smax_pixel[0]-fov[0]:smax_pixel[0]+fov[0],smax_pixel[1]-fov[1]:smax_pixel[1]+fov[1]]/data.max()
max_pixel = np.unravel_index(np.argmax(sdata), sdata.shape)

''' Create the WCS information '''
wcs_dict = {
    'CTYPE1':'HPLT-TAN', ## To use sunpy this needs to be in helioporject coord.
    'CTYPE2':'HPLN-TAN', ## Although strange, it does not affect any of the analysis.
    'CUNIT1':'arcsec',
    'CUNIT2':'arcsec',
    'CDELT1': plate_scale.value, ## Plate scale in arcsec
    'CDELT2': plate_scale.value, ## Plate scale in arcsec
    'CRPIX1':0,
    'CRPIX2':0,
    'CRVAL1': 0,
    'CRVAL2': 0,
    'NAXIS1': sdata.shape[0],
    'NAXIS2': sdata.shape[1]
}
input_wcs = wcs.WCS(wcs_dict)

''' Create NDCube '''
datacube = NDCube(sdata, input_wcs)

''' One 2D-Gaussian '''
''' Definition of One 2D-Gaussian function '''
@models.custom_model
def OneGaussians(x, y,
                 x_mean=0, y_mean=0, theta=0,
                 amp1=0, x1_stddev=0, y1_stddev=0, ## Gauss1 param
                 offset=0): ## offset
    ''' Constrain positive values for the amplitudes '''
    if amp1 < 0:
        amp1 = 1e12

    '''Define Sum of Gauss funtions'''
    g1 = models.Gaussian2D(amp1, x_mean, y_mean, x1_stddev, y1_stddev, theta)
    ''' Defining Offset '''
    oset = models.Const2D(amplitude=offset)
    return g1(x,y) + oset(x,y)

## Make X,Y,Z data for the TWO 2D-Gaussians:
Xg, Yg = np.mgrid[0:datacube.data.shape[0], 0:datacube.data.shape[1]]

''' Fit One 2D-Gaussians '''
## Initial Guess :
OneG_guess = OneGaussians(x_mean=max_pixel[0], y_mean=max_pixel[1], theta=0,
                          amp1=0.8*np.max(datacube.data),
                          x1_stddev=1.5, y1_stddev=2.1,
                          offset=0)

## Finding best fit:
fit1G = fitting.LevMarLSQFitter()
OneG_out = fit1G(OneG_guess, Xg, Yg, datacube.data, maxiter=320) ## maxiter keyword is the max number of iterations.
Zout1G = OneG_out(Xg, Yg)

## Individual Gaussian Functions:
g1OnG = models.Gaussian2D(OneG_out.amp1.value, OneG_out.x_mean.value,
                       OneG_out.y_mean.value, OneG_out.x1_stddev.value,
                       OneG_out.y1_stddev.value, OneG_out.theta.value)
G1OnG = g1OnG(Xg, Yg)
''' Estimate R^2 '''
RS1G = RSquared(datacube.data, fit1G)
'''  Chi Square '''
chisq1 = scipy.stats.chisquare(datacube.data[np.abs(Zout1G) >= 1e-15], f_exp=Zout1G[np.abs(Zout1G) >= 1e-15])

# Create ImageNormalize objects:
normLogT = ImageNormalize(Zout1G, interval=MinMaxInterval(),stretch=LogStretch())
''' Plotting '''
fig, ax1 = plt.subplots(1, 1,figsize=(4.3,4),subplot_kw=dict(projection=datacube.wcs))
im1 = ax1.imshow(Zout1G, origin='lower', cmap=plt.cm.viridis,norm=normLogT)
cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
ax1.set_title('One 2D-GaussFit Log scale',fontsize=14)
plt.show()
''' Print what the amplitud ratios are '''
print('The amplitud for the guessed 2D-Gaussians is: A1 = {0:.5f}.'
      .format(round(OneG_guess.amp1.value,5)))
print('The amplitud for the output 2D-Gaussians is: A1 = {0:.5f}.'
      .format(round(OneG_out.amp1.value,5)))
print('The standard deviation for the One 2D-Gaussians are: S1x = {0:.5f} and S1y = {1:.5f}.'
      .format(round(OneG_out.x1_stddev.value,5),
              round(OneG_out.y1_stddev.value,5),
       ))
print('Offset = {0:.5f}'.format(round(OneG_out.offset.value,5)))
print('$R^2$ for One gaussians is {0:.5f}'.format(round(RS1G,5)))
print('$chi^2$ for one gaussians is {0:.5f}'.format(round(chisq1[0],5)))

## Plot 3D Fitted Gaussian
''' Clipping data '''
fov = [7,7]
sXg = Xg[max_pixel[0]-fov[0]:max_pixel[0]+fov[0],max_pixel[1]-fov[1]:max_pixel[1]+fov[1]]
sYg = Yg[max_pixel[0]-fov[0]:max_pixel[0]+fov[0],max_pixel[1]-fov[1]:max_pixel[1]+fov[1]]
sZout = OneG_out(sXg, sYg)
sG1 = g1OnG(sXg, sYg)
''' 3D plotting '''
fig, ax1 = plt.subplots(1, 1,figsize=(4.3,4),subplot_kw=dict(projection='3d'))
ax1.plot_surface(sXg,sYg,sZout,cmap=plt.cm.viridis)
ax1.set_title('One 2D-GaussFit',fontsize=14)
ax1.set_zlim3d(0, sZout.max())
plt.show()

## Plot Difference Data vs Fit¶ for One 2D-Gaussian
diff = (datacube.data-Zout1G)
diffmax = (np.array(diff.max(),-diff.min())).max()
fov = [15,15]
fig, ax1 = plt.subplots(figsize=(6,6),subplot_kw=dict(projection=datacube.wcs))
im1 = ax1.imshow(diff, origin='lower', cmap=plt.cm.bwr_r,vmin=-diffmax,vmax=diffmax)
cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
ax1.set_xlim(max_pixel[1]-fov[1], max_pixel[1]+fov[1])
ax1.set_ylim(max_pixel[0]-fov[0], max_pixel[0]+fov[0])
ax1.set_title('Difference Data vs Fit - Log scale',fontsize=18)
plt.show()

## Two 2D-Gaussian
''' Definition of Two 2D-Gaussians function '''
@models.custom_model
def TwoGaussians(x, y,
                 x_mean=0, y_mean=0, theta=0,
                 amp1=0, x1_stddev=0, y1_stddev=0, ## Gauss1 param
                 amp2=0, x2_stddev=0, y2_stddev=0, ## Gauss2 param
                 offset=0): ## offset
    ''' Constrain positive values for the amplitudes '''
    if amp1 < 0:
        amp1 = 1e12
    if amp2 < 0:
        amp2 = 1e12

    '''Define Sum of Gauss funtions'''
    g1 = models.Gaussian2D(amp1, x_mean, y_mean, x1_stddev, y1_stddev, theta)
    g2 = models.Gaussian2D(amp2, x_mean, y_mean, x2_stddev, y2_stddev, theta)
    #g3 = models.Gaussian2D(amp3, x_mean, y_mean, x3_stddev, y3_stddev, theta)
    ''' Defining Offset '''
    oset = models.Const2D(amplitude=offset)
    return g1(x,y) + g2(x,y) + oset(x,y)

## Make X,Y,Z data for the TWO 2D-Gaussians:
Xg, Yg = np.mgrid[0:datacube.data.shape[0], 0:datacube.data.shape[1]]

''' Fit Three 2D-Gaussians '''
## Initial Guess :
TwoG_guess = TwoGaussians(x_mean=max_pixel[0], y_mean=max_pixel[1], theta=0,
                          amp1=0.2*np.max(datacube.data),
                          amp2 = 0.8*np.max(datacube.data),
                          x1_stddev=4.0, y1_stddev=3.5,
                          x2_stddev=0.7, y2_stddev=1.7,
                          offset=0)

## Finding best fit:
fit2G = fitting.LevMarLSQFitter()
TwoG_out = fit2G(TwoG_guess, Xg, Yg, datacube.data, maxiter=320) ## maxiter keyword is the max number of iterations.
Zout2G = TwoG_out(Xg, Yg)

## Individual Gaussian Functions:
g1TwG = models.Gaussian2D(TwoG_out.amp1.value, TwoG_out.x_mean.value,
                       TwoG_out.y_mean.value, TwoG_out.x1_stddev.value,
                       TwoG_out.y1_stddev.value, TwoG_out.theta.value)
g2TwG = models.Gaussian2D(TwoG_out.amp2.value, TwoG_out.x_mean.value,
                       TwoG_out.y_mean.value, TwoG_out.x2_stddev.value,
                       TwoG_out.y2_stddev.value, TwoG_out.theta.value)
### Individual Gaussian Arrays:
G1TwG = g1TwG(Xg, Yg)
G2TwG = g2TwG(Xg, Yg)

''' Estimate R^2 '''
RS2G = RSquared(datacube.data, fit2G)
'''  Chi Square '''
chisq2 = scipy.stats.chisquare(datacube.data[np.abs(Zout2G) >= 1e-15], f_exp=Zout2G[np.abs(Zout2G) >= 1e-15])

## Plot Sinlge 2D Gaussian Fit:
# Create ImageNormalize objects:
normLogT = ImageNormalize(Zout2G, interval=MinMaxInterval(),stretch=LogStretch())
normLog1 = ImageNormalize(G1TwG, interval=MinMaxInterval(),stretch=LogStretch())
normLog2 = ImageNormalize(G2TwG, interval=MinMaxInterval(),stretch=LogStretch())
''' Plotting '''
fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(14,4),
                                             subplot_kw=dict(projection=datacube.wcs))
fig.subplots_adjust(wspace = 0.4) ## Sets space between subplots to avoid overlap
## Best fit of the all three gaussians:
im1 = ax1.imshow(Zout2G, origin='lower', cmap=plt.cm.viridis,norm=normLogT)
cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
ax1.set_title('Two 2D-GaussFit Log scale',fontsize=14)
## Gaussian One:
im2 = ax2.imshow(G1TwG, origin='lower', cmap=plt.cm.viridis,norm=normLog1)
cbar2 = fig.colorbar(im2,ax=ax2, fraction=0.046, pad=0.04)
ax2.set_title('2D-Gaussian One - Log scale',fontsize=14)
## Gaussian Two:
im3 = ax3.imshow(G2TwG, origin='lower', cmap=plt.cm.viridis,norm=normLog2)
cbar3 = fig.colorbar(im3,ax=ax3, fraction=0.046, pad=0.04)
ax3.set_title('2D-Gaussian Two - Log scale',fontsize=14)
plt.show()
''' Print what the amplitud ratios are '''
print('The amplitud ratios for the guessed two 2D-Gaussians are: A1 = {0:.5f} and A2 = {1:.5f}.'
      .format(round(TwoG_guess.amp1.value/(TwoG_guess.amp1.value+TwoG_guess.amp2.value),5),
              round(TwoG_guess.amp2.value/(TwoG_guess.amp1.value+TwoG_guess.amp2.value),5),
       ))
print('The amplitud ratios for the two 2D-Gaussians are: A1 = {0:.5f} and A2 = {1:.5f}.'
      .format(round(TwoG_out.amp1.value/(TwoG_out.amp1.value+TwoG_out.amp2.value),5),
              round(TwoG_out.amp2.value/(TwoG_out.amp1.value+TwoG_out.amp2.value),5),
       ))
print('The standard deviation for the two 2D-Gaussians are: S1x = {0:.5f}, S1y = {1:.5f}, S2x = {2:.5f}, and S2y = {3:.5f}.'
      .format(round(TwoG_out.x1_stddev.value,5),
              round(TwoG_out.y1_stddev.value,5),
              round(TwoG_out.x2_stddev.value,5),
              round(TwoG_out.y2_stddev.value,5),
       ))

print('Offset = {0:.5f}'.format(round(TwoG_out.offset.value,5)))
print('$R^2$ for two gaussians is {0:.5f}'.format(round(RS2G,5)))
print('$chi^2$ for two gaussians is {0:.5f}'.format(round(chisq2[0],5)))

## Plot 3D Fitted Gaussians
''' Clipping data '''
fov = [7,7]
sXg = Xg[max_pixel[0]-fov[0]:max_pixel[0]+fov[0],max_pixel[1]-fov[1]:max_pixel[1]+fov[1]]
sYg = Yg[max_pixel[0]-fov[0]:max_pixel[0]+fov[0],max_pixel[1]-fov[1]:max_pixel[1]+fov[1]]
sZout = TwoG_out(sXg, sYg)
sG1 = g1TwG(sXg, sYg)
sG2 = g2TwG(sXg, sYg)

''' 3D plotting '''
fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(14,4),
                                             subplot_kw=dict(projection='3d'))
ax1.plot_surface(sXg,sYg,sZout,cmap=plt.cm.viridis)
ax1.set_title('Two 2D-GaussFit',fontsize=14)
ax1.set_zlim3d(0, sZout.max())
ax2.plot_surface(sXg,sYg,sG1,cmap=plt.cm.viridis)
ax2.set_title('2D-Gaussian One',fontsize=14)
ax2.set_zlim3d(0, sZout.max())
ax3.plot_surface(sXg,sYg,sG2,cmap=plt.cm.viridis)
ax3.set_title('2D-Gaussian Two',fontsize=14)
ax3.set_zlim3d(0, sZout.max())
plt.show()

## Plot Difference Data vs Fit for Two gaussians
diff = (datacube.data-Zout2G)
diffmax = (np.array(diff.max(),-diff.min())).max()
fov = [15,15]
fig, ax1 = plt.subplots(figsize=(6,6),subplot_kw=dict(projection=datacube.wcs))
im1 = ax1.imshow(diff, origin='lower', cmap=plt.cm.bwr_r,vmin=-diffmax,vmax=diffmax)
cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
ax1.set_xlim(max_pixel[1]-fov[1], max_pixel[1]+fov[1])
ax1.set_ylim(max_pixel[0]-fov[0], max_pixel[0]+fov[0])
ax1.set_title('Difference Data vs Fit - Log scale',fontsize=18)
plt.show()
