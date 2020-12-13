"""
----------------------------------------------------------
Estimate the FWHM for FOXSI3-SLF using three 2D-Gaussians
----------------------------------------------------------

Goal: To define a function that calculates the Full Width at Half Max for the
FOXSI3-SLF Data (Corrected by darks) as a function of the Azimuthal angle.
It uses the three 2D-Gaussians fit to estimate such that values.

Input Files:  1. Fits file with SLF Data taken with the Andor CCD Camera.
              2. Dark fits files for correction.

Parameter to set:
    :param binning: CCD Binning
    :param folder: Location of the raw original fits files.
    :param XX: Yaw angle in arcmins
    :param YY: Pitch angle in arcmins
    :param filename: Name of the raw original fits file
    :param SaveFolder: Path where you want to save the outcomes of this function

Run on terminal: ipython PSF_three_2DGaussians_Fit.py

Output:
            1. Map of the Three 2D-Gaussians with the FWHM on top.
            2. Plot of the FWHM vs Azimuthal Angle.
            3. Print out the average over the azimuthal angle of the FWHM.

History:
            Milo created.
UC-Berkeley
"""
# Import Packages:
from ndcube import NDCube
from astropy.modeling import models, fitting
from astropy.io import fits as pyfits
from astropy import wcs
from scipy.optimize import brentq
from datetime import datetime
from astropy.visualization import ImageNormalize, MinMaxInterval, LogStretch
import astropy.units as u
import numpy as np
import scipy.stats
import logging
import os
import matplotlib.pyplot as plt

''' Definition of Two 2D-Gaussians function '''

@models.custom_model
def ThreeGaussians(x, y,
                   x_mean=0, y_mean=0, theta=0,
                   amp1=0, x1_stddev=0, y1_stddev=0,  # Gauss1 param
                   amp2=0, x2_stddev=0, y2_stddev=0,  # Gauss2 param
                   amp3=0, x3_stddev=0, y3_stddev=0,  # Gauss3 param
                   offset=0):  # offset
    ''' Constrain positive values for the amplitudes '''
    if amp1 < 0:
        amp1 = 1e12
    if amp2 < 0:
        amp2 = 1e12
    if amp3 < 0:
        amp3 = 1e12

    '''Define Sum of Gauss funtions'''
    g1 = models.Gaussian2D(amp1, x_mean, y_mean, x1_stddev, y1_stddev, theta)
    g2 = models.Gaussian2D(amp2, x_mean, y_mean, x2_stddev, y2_stddev, theta)
    g3 = models.Gaussian2D(amp3, x_mean, y_mean, x3_stddev, y3_stddev, theta)
    ''' Defining Offset '''
    oset = models.Const2D(amplitude=offset)
    return g1(x, y) + g2(x, y) + g3(x, y) + oset(x, y)

def x_fwhm_minmax(mask, Xt):
    xrange = []
    for xi in range(0, len(Xt)):
        if (mask[xi, :].any() == True):
            xrange.append(Xt[xi, 0])
    return (xrange[0], xrange[-1])

def find_fwhm(G3, datacube, x):  # Input should be a 3-2D-Gaussian Function. e.g. ThreeG_out
    factor = 4 * np.sqrt(2 * np.log(2)) * (G3.y1_stddev +
                                           G3.y2_stddev + G3.y3_stddev)
    steps = 0.5
    ymax = steps * np.argmax([G3(x, yi) for yi in np.arange(0, len(datacube.data), steps)])
    y_fwhm_down = brentq(G3y, ymax - factor, ymax, args=(x, G3))
    y_fwhm_up = brentq(G3y, ymax, ymax + factor, args=(x, G3))
    return (y_fwhm_down, y_fwhm_up)

def G3y(y, x, G3):  # Flip argument order. Needed to find zeros on y-axis.
    return G3(x, y)

def y_fwhm_minmax(mask, Yt):
    yrange = []
    for yi in range(0, len(Yt)):
        if (mask[:, yi].any() == True):
            yrange.append(Yt[0, yi])
    return (yrange[0], yrange[-1])

def find_fwhmY(G3, datacube, y):  # Input should be a 3-2D-Gaussian Function. e.g. ThreeG_out
    factor = 4 * np.sqrt(2 * np.log(2)) * (G3.x1_stddev +
                                           G3.x2_stddev + G3.x3_stddev)
    steps = 0.5
    xmax = steps * np.argmax([G3(xi, y) for xi in np.arange(0, len(datacube.data), steps)])
    x_fwhm_left = brentq(G3, xmax - factor, xmax, args=(y, G3))
    x_fwhm_right = brentq(G3, xmax, xmax + factor, args=(y, G3))
    return (x_fwhm_left, x_fwhm_right)

# -------------------------------------------------
#                  MAIN PROGRAM
# -------------------------------------------------

XX = 0.0  # Yaw in arcmin
YY = 0.0  # Pitch in arcmin

# Path to the folder where to find your data and darks:
folder = './data/'
SaveFolder = './'

# File names:
filename = folder+'LabData.fits'  # name of your data fits file.
darkfilename = folder+'Darks.fits'  # name of your darks fits file.
# These are fits files containing six frames each of 1024x1024 pixels taken at the SLF
# using the Andor camera and the Mo X-ray source. Voltages,Currents and Integration Times are
# indicated over the names of the files.

# LogFile Creation:
if os.path.exists(SaveFolder + 'LogFile.log'):
    os.remove(SaveFolder + 'LogFile.log')
logging.basicConfig(filename=SaveFolder + 'LogFile_'+datetime.now().strftime('%Y%m%d-%H%M')+'.log', level=logging.INFO)
logging.info('LogFile - '+datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


# Plate scale
distance = 2. * u.m  # FOXSI focal distance
CCD_PIXEL_PITCH = 13.5 * u.micron  # Andor CCD Pitch in microns
binning = 2.  # binning used for collecting data.
plate_scale = np.arctan(binning * CCD_PIXEL_PITCH / distance).to(u.arcsec)  # in arcsec

# Read fits files using astropy.io.fits
fits = pyfits.open(filename)
darkfits = pyfits.open(darkfilename)

# Create data array corrected by darks:
data = np.average(fits[0].data, axis=0) - np.average(darkfits[0].data, axis=0)
smax_pixel = np.unravel_index(np.argmax(data), data.shape)
fov = [int(36 / binning), int(36 / binning)]  # [px,px]
sdata = data[smax_pixel[0] - fov[0]:smax_pixel[0] + fov[0],
        smax_pixel[1] - fov[1]:smax_pixel[1] + fov[1]] / data.max()
max_pixel = np.unravel_index(np.argmax(sdata), sdata.shape)

''' Create the WCS information '''
wcs_dict = {
    'CTYPE1': 'HPLT-TAN',  # To use sunpy this needs to be in helioproject coord.
    'CTYPE2': 'HPLN-TAN',  # Although strange, it does not affect any of the analysis.
    'CUNIT1': 'arcsec',
    'CUNIT2': 'arcsec',
    'CDELT1': plate_scale.value,  # Plate scale in arcsec
    'CDELT2': plate_scale.value,  # Plate scale in arcsec
    'CRPIX1': max_pixel[0]+1,
    'CRPIX2': max_pixel[1]+1,
    'CRVAL1': 0,
    'CRVAL2': 0,
    'NAXIS1': sdata.shape[0],
    'NAXIS2': sdata.shape[1]
}
input_wcs = wcs.WCS(wcs_dict)

''' Create NDCube '''
datacube = NDCube(sdata, input_wcs)

# Make X,Y,Z data for the THREE 2D-Gaussians:
Xg, Yg = np.mgrid[0:datacube.data.shape[0], 0:datacube.data.shape[1]]

# Fit Three 2D-Gaussians
# Initial Guess :
ThreeG_guess = ThreeGaussians(x_mean=max_pixel[0], y_mean=max_pixel[1], theta=0,
                              amp1=0.10111 * np.max(datacube.data),
                              amp2=0.57882 * np.max(datacube.data),
                              amp3=0.32008 * np.max(datacube.data),
                              x1_stddev=5.0171, y1_stddev=4.0530,
                              x2_stddev=0.6243, y2_stddev=1.2561,
                              x3_stddev=1.5351, y3_stddev=2.2241, offset=0)

# Finding best fit:
fit2DG = fitting.LevMarLSQFitter()
ThreeG_out = fit2DG(ThreeG_guess, Xg, Yg, datacube.data, maxiter=320)
# maxiter keyword is the max number of iterations.
Zout = ThreeG_out(Xg, Yg)

'''  Chi Square '''
chisq = scipy.stats.chisquare(datacube.data[np.abs(Zout) >= 1e-15], f_exp=Zout[np.abs(Zout) >= 1e-15])

''' Print what the amplitud ratios are '''
logging.info('*' * 30)
logging.info('Parameters for {0} Yaw & {1} Pitch : '.format(str(XX), str(YY)))
logging.info('The amplitud ratios for the guessed three 2D-Gaussians are: A1 = {0:.5f}, A2 = {1:.5f}, and {2:.5f}.'
    .format(
    round(ThreeG_guess.amp1.value / (ThreeG_guess.amp1.value + ThreeG_guess.amp2.value + ThreeG_guess.amp3.value),
          5),
    round(ThreeG_guess.amp2.value / (ThreeG_guess.amp1.value + ThreeG_guess.amp2.value + ThreeG_guess.amp3.value),
          5),
    round(ThreeG_guess.amp3.value / (ThreeG_guess.amp1.value + ThreeG_guess.amp2.value + ThreeG_guess.amp3.value),
          5),
))
logging.info('The amplitud ratios for the three 2D-Gaussians are: A1 = {0:.5f}, A2 = {1:.5f}, and {2:.5f}.'
    .format(
    round(ThreeG_out.amp1.value / (ThreeG_out.amp1.value + ThreeG_out.amp2.value + ThreeG_out.amp3.value), 5),
    round(ThreeG_out.amp2.value / (ThreeG_out.amp1.value + ThreeG_out.amp2.value + ThreeG_out.amp3.value), 5),
    round(ThreeG_out.amp3.value / (ThreeG_out.amp1.value + ThreeG_out.amp2.value + ThreeG_out.amp3.value), 5),
))
logging.info(
    'The standard deviation for the three 2D-Gaussians are: S1x = {0:.5f}, S1y = {1:.5f}, S2x = {2:.5f}, S2y = {3:.5f}, S3x = {4:.5f}, and S3y = {5:.5f}.'
        .format(round(ThreeG_out.x1_stddev.value, 5),
                round(ThreeG_out.y1_stddev.value, 5),
                round(ThreeG_out.x2_stddev.value, 5),
                round(ThreeG_out.y2_stddev.value, 5),
                round(ThreeG_out.x3_stddev.value, 5),
                round(ThreeG_out.y3_stddev.value, 5),
                ))
phi = np.rad2deg(ThreeG_out.theta.value - (ThreeG_out.theta.value // np.pi) * np.pi)
logging.info('Offset = {0:.5f}'.format(round(ThreeG_out.offset.value, 5)))
logging.info('Angle = {0:.5f} degrees'.format(round(phi, 3)))
logging.info('$chi^2$ for three gaussians is {0:.5f}'.format(round(chisq[0] / (Zout.shape[0] * Zout.shape[1]), 5)))

# Estimate of the FWHM on X&Y
maximum = ThreeG_out.amp1.value \
          + ThreeG_out.amp2.value \
          + ThreeG_out.amp3.value
half_maximum = 0.5 * maximum

ThreeG_out.offset -= half_maximum
npoints = 50

# Needed to determine the size of the FWHM:
steps = 78j  # Play with this value to close the FWHM or to make it converge.
Xt, Yt = np.mgrid[0:datacube.data.shape[0]:steps, 0:datacube.data.shape[1]:steps]
ZoutHR = ThreeG_out(Xt, Yt)  # increasing the resolution of Zout
mask = np.greater(ZoutHR, 0)
x_fwhm = ThreeG_out.x_mean + (ThreeG_out.x_mean -
                              x_fwhm_minmax(mask, Xt)[0]) * np.sin((np.pi / 2) * np.linspace(-1, 1, npoints))
y_fwhm_up_l, y_fwhm_down_l = [], []
for x in x_fwhm:
    y_fwhm = find_fwhm(ThreeG_out, datacube, x)
    y_fwhm_down_l.append(y_fwhm[0])
    y_fwhm_up_l.append(y_fwhm[1])
y_fwhm_down = np.array(y_fwhm_down_l)
y_fwhm_up = np.array(y_fwhm_up_l)

# Mapping along the vertical axis
y_fwhm2 = ThreeG_out.y_mean + (ThreeG_out.y_mean -
                               y_fwhm_minmax(mask, Yt)[0]) * np.sin((np.pi / 2) * np.linspace(-1, 1, npoints))
x_fwhm_right_l, x_fwhm_left_l = [], []
for y in y_fwhm2:
    x_fwhm2 = find_fwhmY(ThreeG_out, datacube, y)
    x_fwhm_left_l.append(x_fwhm2[0])
    x_fwhm_right_l.append(x_fwhm2[1])
x_fwhm_left = np.array(x_fwhm_left_l)
x_fwhm_right = np.array(x_fwhm_right_l)

ThreeG_out.offset += half_maximum

r_up = np.sqrt((x_fwhm - ThreeG_out.x_mean) ** 2 + (y_fwhm_up - ThreeG_out.y_mean) ** 2)
r_down = np.sqrt((x_fwhm - ThreeG_out.x_mean) ** 2 + (y_fwhm_down - ThreeG_out.y_mean) ** 2)
phi_up = np.arctan2((x_fwhm - ThreeG_out.x_mean), (y_fwhm_up - ThreeG_out.y_mean))
phi_down = np.arctan2((x_fwhm - ThreeG_out.x_mean), (y_fwhm_down - ThreeG_out.y_mean))
r_right = np.sqrt((x_fwhm_right - ThreeG_out.x_mean) ** 2 + (y_fwhm2 - ThreeG_out.y_mean) ** 2)
r_left = np.sqrt((x_fwhm_left - ThreeG_out.x_mean) ** 2 + (y_fwhm2 - ThreeG_out.y_mean) ** 2)
phi_right = np.arctan2((x_fwhm_right - ThreeG_out.x_mean), (y_fwhm2 - ThreeG_out.y_mean))
phi_left = np.arctan2((x_fwhm_left - ThreeG_out.x_mean), (y_fwhm2 - ThreeG_out.y_mean))

r = np.concatenate((r_up, r_down, r_right, r_left))
phis = np.concatenate((phi_up, phi_down, phi_right, phi_left))

logging.info('FWHM for {0} arcmin in Yaw and {1} arcmin in Pitch:'.format(str(XX), str(YY)))
logging.info('The average FWHM over the azimuthal angle is {0} arcsecs.'.format(round(2 * r.mean() * plate_scale.value, 4)))

''' Plotting '''
# Create ImageNormalize objects:
normLogT = ImageNormalize(datacube.data, interval=MinMaxInterval(), stretch=LogStretch())
fig1, axs = plt.subplots(1, 2,figsize=(12, 6), subplot_kw=dict(projection=datacube.wcs))
# Plot of everything together [contours]:
im1 = axs[0].imshow(datacube.data, origin='lower', cmap=plt.cm.viridis, norm=normLogT, vmin=0.0, vmax=1.0)
axs[0].scatter(y_fwhm_down, x_fwhm, c='blue', s=3)
axs[0].scatter(y_fwhm_up, x_fwhm, c='blue', s=3)
axs[0].scatter(y_fwhm2, x_fwhm_left, c='blue', s=3)
axs[0].scatter(y_fwhm2, x_fwhm_right, c='blue', s=3)
axs[0].set_title('Lab-Data (black), Fit (white) & FWHM (blue) on top of Lab-PSF', fontsize=12)
levels = np.array([.01, .1, .25, .5, .75, .85])  # Set level at half the maximum
CFWHM_dat = axs[0].contour(datacube.data, levels, colors='black')  # Generate contour Data
CFWHM_fit = axs[0].contour(Zout, levels, colors='white')  # Generate contour Fit
cbar1 = fig1.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)
cbar1.add_lines(CFWHM_fit)
cbar1.ax.plot([0, 1], [0.9, 0.9], c='blue')
# Differences Plot
diff = (datacube.data - Zout)
diffmax = (np.array(diff.max(), -diff.min())).max()
im2 = axs[1].imshow(diff, origin='lower', cmap=plt.cm.bwr_r, vmin=-diffmax, vmax=diffmax)
fig1.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)
axs[1].set_title('Difference Lab-Data vs Fit', fontsize=18)
fig1.suptitle('Yaw {0} arcmin & Pitch {1} arcmin'.format(str(XX), str(YY)), fontsize=18)
plt.savefig(SaveFolder + '{0}Yaw&{1}Pitch_Contours.pdf'.format(str(XX), str(YY)), transparent=True,bbox_inches='tight')
plt.close(fig1)

# FWHM vs Angle Plot

fig1 = plt.figure(figsize=(12, 6))
ax1 = fig1.add_subplot(121)
ax1.plot(np.rad2deg(phis) + phi, 2 * r * plate_scale, 'o')
ax1.set_xlim(0, 90)
ax1.axhline(round(r.max() * 2 * plate_scale.value, 3), linestyle='dashed', color='gray', label='Max = {0}"'.format(round(r.max() * 2 * plate_scale.value, 3)))
ax1.axhline(round((r.max()+r.min()) * 0.5 * 2 * plate_scale.value, 3), linestyle='dashed', color='#1f77b4', label='Mean = {0}"'.format(round((r.max()+r.min()) * 0.5 * 2 * plate_scale.value, 3)))
ax1.axhline(round(r.min() * 2 * plate_scale.value, 3), linestyle='dashed', color='gray', label='Min = {0}"'.format(round(r.min() * 2 * plate_scale.value, 3)))
ax1.xaxis.set_tick_params(labelsize=16)
ax1.yaxis.set_tick_params(labelsize=16)
ax1.set_xlabel('Azimuthal angle [degrees]', fontsize=20)
ax1.set_ylabel('FWHM [arcsec]', fontsize=20)
ax1.set_ylim(0, 15)
leg = ax1.legend(loc='best', prop={'size': 16})
for artist, text in zip(leg.legendHandles, leg.get_texts()):
    try:
        col = artist.get_color()
    except:
        col = artist.get_facecolor()
    if isinstance(col, np.ndarray):
        col = col[0]
    text.set_color(col)
wcs_dict['CRPIX1'] = ThreeG_out.y_mean.value + 1
wcs_dict['CRPIX2'] = ThreeG_out.x_mean.value + 1
input_wcs = wcs.WCS(wcs_dict)
ax2 = fig1.add_subplot(122, projection=input_wcs)
ax2.scatter(y_fwhm_down, x_fwhm, c='#1f77b4', s=3)
ax2.scatter(y_fwhm_up, x_fwhm, c='#1f77b4', s=3)
ax2.scatter(y_fwhm2, x_fwhm_left, c='#1f77b4', s=3)
ax2.scatter(y_fwhm2, x_fwhm_right, c='#1f77b4', s=3)
fov2 = 8 / binning
ax2.set_xlim(ThreeG_out.y_mean.value-fov2, ThreeG_out.y_mean.value+fov2)
ax2.set_ylim(ThreeG_out.x_mean.value-fov2, ThreeG_out.x_mean.value+fov2)
ax2.axhline(ThreeG_out.x_mean.value, linestyle='dotted', linewidth=1, color='gray')
ax2.axvline(ThreeG_out.y_mean.value, linestyle='dotted', linewidth=1, color='gray')
for r_circle in np.arange(1, 15):
    circle_i = plt.Circle((ThreeG_out.y_mean.value, ThreeG_out.x_mean.value), r_circle / plate_scale.value, color='gray', linestyle='dotted', linewidth=0.5, fill=False)
    ax2.add_artist(circle_i)
    ax2.annotate(str(r_circle), xy=(.02+ThreeG_out.y_mean.value, .02+ThreeG_out.x_mean.value+r_circle/plate_scale.value), fontsize=6, color='gray')
fig1.suptitle('FWHM vs azimuthal angle for Yaw {0} arcmin & Pitch {1} arcmin'.format(str(XX), str(YY)), fontsize=18)
fig1.savefig(SaveFolder + '{0}Yaw&{1}Pitch_FWHMvsAngle.pdf'.format(XX, YY),bbox_inches='tight')
plt.close(fig1)
