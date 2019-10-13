"""
---------------------------------------------------
Simple way to fit Three 2D-Gaussians for FOXSI3-SLF Data (Dark corrected)
Then get the FWHM as a function of the azimuthal angle
---------------------------------------------------

Goal: To fit Three 2D gaussians to the FOXSI3 SLF data corrected by darks.

Input:  1. Fits file with SLF Data taken with the Andor CCD Camera.
        2. Dark fits files for correction.
        3. folder path: Where all FITS files are located.
        4. SaveFolder Path: Folder where to save all images.


Run on terminal: ipython 01_FWHM_OffAxis.py

Output:
            1. Figure including
                i. Lab-PSF in Log scale with contours on top.
                ii. Map differences between Lab-Data and best Fit.
            2. Figure of the FWHM as a function of the azimuthal angle in
                i. A plot from 0 to 90 degrees.
                ii. Contours based on best fit.
            3. LogFile of ALL the parameters for best fit.
            4. Compiled Plot of the ALL FWHMs as a function of the Off-axis angles [FWHM_vs_OffAxis.png].

Date: Oct, 2019
Author: Milo
UC-Berkeley
"""

# Import Packages:
from ndcube import NDCube
from astropy.modeling import models, fitting
from astropy.visualization import ImageNormalize, MinMaxInterval, LogStretch
from astropy import wcs
from astropy.io import fits as pyfits
from scipy.optimize import brentq
from datetime import datetime
import astropy.units as u
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import glob
import os
import logging
import pdb

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


def RSquared(data, model):
    """ The closest to 1, the better is the fit.
        More info: https://en.wikipedia.org/wiki/Coefficient_of_determination
    """
    ss_err = (model.fit_info['fvec'] ** 2).sum()
    ss_tot = ((data - data.mean()) ** 2).sum()
    return 1 - (ss_err / ss_tot)


def x_fwhm_minmax(mask, Xt):
    xrange = []
    for xi in range(0, len(Xt)):
        if (mask[xi, :].any() == True):
            xrange.append(Xt[xi, 0])
    return (xrange[0], xrange[-1])


def G3y(y, x, G3):  # Flip argument order. Needed to find zeros on y-axis.
    return G3(x, y)


def find_fwhm(G3, datacube, x):  # Input should be a 3-2D-Gaussian Function. e.g. ThreeG_out
    factor = 4 * np.sqrt(2 * np.log(2)) * (G3.y1_stddev +
                                           G3.y2_stddev + G3.y3_stddev)
    steps = 0.5
    ymax = steps * np.argmax([G3(x, yi) for yi in np.arange(0, len(datacube.data), steps)])
    y_fwhm_down = brentq(G3y, ymax - factor, ymax, args=(x, G3))
    y_fwhm_up = brentq(G3y, ymax, ymax + factor, args=(x, G3))
    return (y_fwhm_down, y_fwhm_up)


def find_fwhmY(G3, datacube, y):  # Input should be a 3-2D-Gaussian Function. e.g. ThreeG_out
    factor = 4 * np.sqrt(2 * np.log(2)) * (G3.x1_stddev +
                                           G3.x2_stddev + G3.x3_stddev)
    steps = 0.5
    xmax = steps * np.argmax([G3(xi, y) for xi in np.arange(0, len(datacube.data), steps)])
    x_fwhm_left = brentq(G3, xmax - factor, xmax, args=(y, G3))
    x_fwhm_right = brentq(G3, xmax, xmax + factor, args=(y, G3))
    return (x_fwhm_left, x_fwhm_right)


def y_fwhm_minmax(mask, Yt):
    yrange = []
    for yi in range(0, len(Yt)):
        if (mask[:, yi].any() == True):
            yrange.append(Yt[0, yi])
    return (yrange[0], yrange[-1])


def FWHMvsAnglePlot(XX, YY, phi, phis, r, plate_scale, wcs_dict, SaveFolder, x_mean, y_mean,
                    y_fwhm_down, y_fwhm_up, x_fwhm, y_fwhm2, x_fwhm_left, x_fwhm_right):
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
    wcs_dict['CRPIX1'] = y_mean.value + 1
    wcs_dict['CRPIX2'] = x_mean.value + 1
    input_wcs = wcs.WCS(wcs_dict)
    ax2 = fig1.add_subplot(122, projection=input_wcs)
    ax2.scatter(y_fwhm_down, x_fwhm, c='#1f77b4', s=3)
    ax2.scatter(y_fwhm_up, x_fwhm, c='#1f77b4', s=3)
    ax2.scatter(y_fwhm2, x_fwhm_left, c='#1f77b4', s=3)
    ax2.scatter(y_fwhm2, x_fwhm_right, c='#1f77b4', s=3)
    fov2 = 4
    ax2.set_xlim(y_mean.value-fov2, y_mean.value+fov2)
    ax2.set_ylim(x_mean.value-fov2, x_mean.value+fov2)
    ax2.axhline(x_mean.value, linestyle='dotted', linewidth=1, color='gray')
    ax2.axvline(y_mean.value, linestyle='dotted', linewidth=1, color='gray')
    for r_circle in np.arange(1, 15):
        circle_i = plt.Circle((y_mean.value, x_mean.value), r_circle / plate_scale.value, color='gray', linestyle='dotted', linewidth=0.5, fill=False)
        ax2.add_artist(circle_i)
        ax2.annotate(str(r_circle), xy=(.02+y_mean.value, .02+x_mean.value+r_circle/plate_scale.value), fontsize=6, color='gray')
    fig1.suptitle('FWHM vs azimuthal angle for Yaw {0} arcmin & Pitch {1} arcmin'.format(str(XX), str(YY)), fontsize=18)
    fig1.savefig(SaveFolder + '{0}Yaw&{1}Pitch_FWHMvsAngle.pdf'.format(XX, YY))
    plt.close(fig1)
    return 2*r*plate_scale.value

def RunAll(XX, YY, filename, SaveFolder):
    """
    :param folder: Location of the raw original fits files.
    :param XX: Yaw angle in arcmins
    :param YY: Pitch angle in arcmins
    :param filename: Name of the raw original fits file
    :param SaveFolder: Path where you want to save the outcomes of this function
    :return: X & Y arrays of the FWHM.
    """
    XX = XX  # Yaw in arcmin
    YY = YY  # Pitch in arcmin

    print('Yaw = {0} arcmin, Pitch = {1}'.format(str(XX), str(YY)))

    # File names:
    filename = filename
    darkfilename = '/Volumes/Pandora/FOXSI/OpCal/FOXSI-3_2018Mar/X1-7Shells_NewBlockers/CCD/Darks_X1_NewBlockers/Darks1_FOXSI3_X1_NewBlockers_CCD_T9Sx6_10kV_0p02mA_+6p4arcminX_-6p4arcminY.fits'  ## name of your darks fits file.

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

    # Make X,Y,Z data for the TWO 2D-Gaussians:
    Xg, Yg = np.mgrid[0:datacube.data.shape[0], 0:datacube.data.shape[1]]

    ''' Fit Three 2D-Gaussians '''
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

    ''' Estimate R^2 '''
    RS3G = RSquared(datacube.data, fit2DG)
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
    logging.info('$R^2$ for three gaussians is {0:.5f}'.format(round(RS3G, 5)))
    logging.info('$chi^2$ for three gaussians is {0:.5f}'.format(round(chisq[0] / (Zout.shape[0] * Zout.shape[1]), 5)))

    # Estimate of the FWHM on X&Y
    maximum = ThreeG_out.amp1.value \
              + ThreeG_out.amp2.value \
              + ThreeG_out.amp3.value
    half_maximum = 0.5 * maximum
    ThreeG_out.offset -= half_maximum
    npoints = 50

    # Needed to determine the size of the FWHM:
    steps = 60j  # Play with this value to close the FWHM or to make it converge.
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
    # Plot FWHM vs Angle:
    FWHMr = FWHMvsAnglePlot(XX, YY, phi, phis, r, plate_scale, wcs_dict, SaveFolder, ThreeG_out.x_mean, ThreeG_out.y_mean,
                            y_fwhm_down, y_fwhm_up, x_fwhm, y_fwhm2, x_fwhm_left, x_fwhm_right)
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
    levels = np.array([.01, .1, .25, .5, .75, .9])  # Set level at half the maximum
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
    plt.savefig(SaveFolder + '{0}Yaw&{1}Pitch_Contours.pdf'.format(str(XX), str(YY)), transparent=True)
    plt.close(fig1)

    logging.info('FWHM for {0} arcmin in Yaw and {1} arcmin in Pitch:'.format(str(XX), str(YY)))
    logging.info('The average FWHM over the azimuthal angle is {0} arcsecs.'.format(round(2 * r.mean() * plate_scale.value, 4)))

    XX_fwhm = np.concatenate((y_fwhm_down, y_fwhm_up, y_fwhm2, y_fwhm2), axis=0)
    YY_fwhm = np.concatenate((x_fwhm, x_fwhm, x_fwhm_left, x_fwhm_right), axis=0)

    # Creates list with fit contours for each off-axis angle.
    CX_Dat, CY_Dat = [], []
    for c in CFWHM_dat.collections:
        v = c.get_paths()[0].vertices
        CX_Dat.append(v[:, 0] - ThreeG_out.x_mean.value)
        CY_Dat.append(v[:, 1] - ThreeG_out.y_mean.value)

    CX_Fit, CY_Fit = [], []
    for c in CFWHM_fit.collections:
        v = c.get_paths()[0].vertices
        CX_Fit.append(v[:, 0] - ThreeG_out.x_mean.value)
        CY_Fit.append(v[:, 1] - ThreeG_out.y_mean.value)

    return [np.array([XX_fwhm - ThreeG_out.y_mean.value, YY_fwhm - ThreeG_out.x_mean.value]),
            ((CX_Dat, CY_Dat), (CX_Fit, CY_Fit)), FWHMr]


''' Main program '''
# Path to the folder where to save all the outcomes:
SaveFolder = '/Users/Kamilobu/Desktop/test/'

# LogFile Creation:
if os.path.exists(SaveFolder + 'LogFile.log'):
    os.remove(SaveFolder + 'LogFile.log')
logging.basicConfig(filename=SaveFolder + 'LogFile_'+datetime.now().strftime('%Y%m%d-%H%M')+'.log', level=logging.INFO)
logging.info('LogFile - '+datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

# Looping over the whole set of data.
folder = '/Volumes/Pandora/FOXSI/OpCal/FOXSI-3_2018Mar/X1-7Shells_NewBlockers/CCD/PSF/'
str_indices_vh = {'0p5a': '', '1a': '', '2a': '', '3a': '', '5a': '', '7a': '', '9a': ''}
str_indices_ds = {'0p4a': '', '0p7a': '', '1p4a': '', '2p1a': '', '3p5a': '', '4p9a': '', '6p4a': ''}

flist_000, flist_090, flist_045, flist_135 = [], [], [], []
for key in str_indices_vh:
    flist_000.append(glob.glob(folder + '*+' + key + '*0a*.fits')[0])
    flist_000.append(glob.glob(folder + '*-' + key + '*0a*.fits')[0])
    flist_090.append(glob.glob(folder + '*0a*+' + key + '*.fits')[0])
    flist_090.append(glob.glob(folder + '*0a*-' + key + '*.fits')[0])

for key in str_indices_ds:
    flist_045.append(glob.glob(folder + '*+' + key + '*+' + key + '*.fits')[0])
    flist_045.append(glob.glob(folder + '*-' + key + '*-' + key + '*.fits')[0])
    flist_135.append(glob.glob(folder + '*+' + key + '*-' + key + '*.fits')[0])
    flist_135.append(glob.glob(folder + '*-' + key + '*+' + key + '*.fits')[0])

nlist_000_XX = [0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 5.0, -5.0, 7.0, -7.0, 9.0, -9.0]
nlist_045_XX = [0.4, -0.4, 0.7, -0.7, 1.4, -1.4, 2.1, -2.1, 3.5, -3.5, 4.9, -4.9, 6.4, -6.4]

FWHM_all, CDat_all, CFit_all, XX_all, YY_all, R_all = [], [], [], [], [], []
# On axis
filename = '/Volumes/Pandora/FOXSI/OpCal/FOXSI-3_2018Mar/X1-7Shells_NewBlockers/CCD/PSF/FOXSI3_X1_NewBlockers_CCD_T9Sx6_10kV_0p02mA_0arcminX_0arcminY.fits'
XX_all.append(0.0)
YY_all.append(0.0)
FWHMi, (CDat, CFit), FWHMr = RunAll(0.0, 0.0, filename, SaveFolder)
FWHM_all.append(FWHMi)
CDat_all.append(CDat)
CFit_all.append(CFit)
R_all.append(FWHMr)

# 000 - The negative sign for XX is due to the mirroring flip of the CCD.
for XX, filename in zip(nlist_000_XX, flist_000):
    XX_all.append(-XX)
    YY_all.append(0.0)
    FWHMi, (CDat, CFit), FWHMr = RunAll(-XX, 0.0, filename, SaveFolder)
    FWHM_all.append(FWHMi)
    CDat_all.append(CDat)
    CFit_all.append(CFit)
    R_all.append(FWHMr)

# 090
for YY, filename in zip(nlist_000_XX, flist_090):
    XX_all.append(0.0)
    YY_all.append(YY)
    FWHMi, (CDat, CFit), FWHMr = RunAll(0.0, YY, filename, SaveFolder)
    FWHM_all.append(FWHMi)
    CDat_all.append(CDat)
    CFit_all.append(CFit)
    R_all.append(FWHMr)

# 045
for XX, YY, filename in zip(nlist_045_XX, nlist_045_XX, flist_045):
    XX_all.append(-XX)
    YY_all.append(YY)
    FWHMi, (CDat, CFit), FWHMr = RunAll(-XX, YY, filename, SaveFolder)
    FWHM_all.append(FWHMi)
    CDat_all.append(CDat)
    CFit_all.append(CFit)
    R_all.append(FWHMr)

# 135
for XX, YY, filename in zip(nlist_045_XX, nlist_045_XX, flist_135):
    XX_all.append(-XX);
    YY_all.append(-YY)
    FWHMi, (CDat, CFit), FWHMr = RunAll(-XX, -YY, filename, SaveFolder)
    FWHM_all.append(FWHMi)
    CDat_all.append(CDat)
    CFit_all.append(CFit)
    R_all.append(FWHMr)

# # Next three lines for testing:
# filename = '/Volumes/Pandora/FOXSI/OpCal/FOXSI-3_2018Mar/X1-7Shells_NewBlockers/CCD/PSF/FOXSI3_X1_NewBlockers_CCD_T9Sx6_10kV_0p02mA_0arcminX_0arcminY.fits'
# XX_all.append(0.0)
# YY_all.append(0.0)
# FWHMi, (CDat, CFit) = RunAll(0.0, 0.0, filename, SaveFolder)
# FWHM_all.append(FWHMi)
# CDat_all.append(CDat)
# CFit_all.append(CFit)
# R_all.append(FWHMr)

''' Plot of the FWHMs as function of the off-axis angles - with a factor of 6 '''
Amplification = 0.2
scale = 60.*Amplification  # this is 5 times larger than the actual size of the FWHM.
fig1, ax1 = plt.subplots(figsize=(80, 80))
for c in nlist_000_XX:
    circle = plt.Circle((0, 0), c, color='gray', linestyle='dashed', fill=False)
    ax1.add_artist(circle)
for i in range(0, len(FWHM_all)):
    ax1.scatter(FWHM_all[i][0] / scale + XX_all[i], FWHM_all[i][1] / scale + YY_all[i], c='blue', s=8)
ax1.tick_params(labelsize=80, length=20, width=5)
ax1.set_xlabel('Yaw [arcmin]', fontsize=80)
ax1.set_ylabel('Pitch [arcmin]', fontsize=80)
ax1.set_title('FWHM as a function of Off-axis angles [{0}x Amplified]'.format(round(1/Amplification, 0)), fontsize=140)
ax1.set_ylim(-10, 10)
ax1.set_xlim(-10, 10)
plt.savefig(SaveFolder + 'FWHM_vs_OffAxis.pdf')
plt.close(fig1)

# Plot of the Data Contours as function of the off-axis angles - with a factor of 6
scale = 60.  # this is 60*0.05 = 3 times the actual size of the contours
fig1, ax1 = plt.subplots(figsize=(80, 80))
for c in nlist_000_XX:
    circle = plt.Circle((0, 0), c, color='gray', linestyle='dashed', fill=False)
    ax1.add_artist(circle)
for i in range(0, len(CDat_all)):
    for j in range(0, len(CDat_all[0][0])):
        ax1.plot(np.array(CDat_all[i][0][j] / scale) + XX_all[i], np.array(CDat_all[i][1][j] / scale) + YY_all[i],
                 color='r')
ax1.tick_params(labelsize=80, length=20, width=5)
ax1.set_xlabel('Yaw [arcmin]', fontsize=80)
ax1.set_ylabel('Pitch [arcmin]', fontsize=80)
ax1.set_title('Data Contours as a function of Off-axis angles', fontsize=140)
ax1.set_ylim(-10, 10)
ax1.set_xlim(-10, 10)
fig1.savefig(SaveFolder + 'DatCont_vs_OffAxis.pdf')
plt.close(fig1)

''' Plot of the FWHMs as function of the off-axis angles - with a factor of 6 '''
scale = 60.
fig2, ax2 = plt.subplots(figsize=(80, 80))
for c in nlist_000_XX:
    circle = plt.Circle((0, 0), c, color='gray', linestyle='dashed', fill=False)
    ax2.add_artist(circle)
for i in range(0, len(CFit_all)):
    for j in range(0, len(CFit_all[0][0])):
        ax2.plot(np.array(CFit_all[i][0][j] / scale) + XX_all[i], np.array(CFit_all[i][1][j] / scale) + YY_all[i],
                 color='r')
ax2.tick_params(labelsize=80, length=20, width=5)
ax2.set_xlabel('Yaw [arcmin]', fontsize=80)
ax2.set_ylabel('Pitch [arcmin]', fontsize=80)
ax2.set_ylim(-10, 10)
ax2.set_xlim(-10, 10)
ax2.set_title('Fit Contours as a function of Off-axis angles', fontsize=140)
fig2.savefig(SaveFolder + 'FitCont_vs_OffAxis.pdf')
plt.close(fig2)


''' Plot of Min & Max of the FWHMs as function of the off-axis angles '''

fwhm_mean,fwhm_min,fwhm_max = [],[],[]
for f in R_all:
    fwhm_mean.append((f.max()+f.min())/2.)
    fwhm_min.append(f.min())
    fwhm_max.append(f.max())

Mean000 = np.take_along_axis(np.array(fwhm_mean[0:15]),np.argsort(XX_all[0:15], axis=0),axis=0)
Min000  = np.take_along_axis(np.array(fwhm_min[0:15]),np.argsort(XX_all[0:15], axis=0),axis=0)
Max000  = np.take_along_axis(np.array(fwhm_max[0:15]),np.argsort(XX_all[0:15], axis=0),axis=0)

Mean090 = np.take_along_axis(np.array(fwhm_mean[15:29]),np.argsort(YY_all[15:29], axis=0),axis=0)
Min090  = np.take_along_axis(np.array(fwhm_min[15:29]),np.argsort(YY_all[15:29], axis=0),axis=0)
Max090  = np.take_along_axis(np.array(fwhm_max[15:29]),np.argsort(YY_all[15:29], axis=0),axis=0)

Mean045 = np.take_along_axis(np.array(fwhm_mean[29:43]),np.argsort(XX_all[29:43], axis=0),axis=0)
Min045  = np.take_along_axis(np.array(fwhm_min[29:43]),np.argsort(XX_all[29:43], axis=0),axis=0)
Max045  = np.take_along_axis(np.array(fwhm_max[29:43]),np.argsort(XX_all[29:43], axis=0),axis=0)

Mean135 = np.take_along_axis(np.array(fwhm_mean[43:57]),np.argsort(XX_all[43:57], axis=0),axis=0)
Min135  = np.take_along_axis(np.array(fwhm_min[43:57]),np.argsort(XX_all[43:57], axis=0),axis=0)
Max135  = np.take_along_axis(np.array(fwhm_max[43:57]),np.argsort(XX_all[43:57], axis=0),axis=0)

FMean = np.insert((np.delete(Mean000, 7) + Mean090 + Mean045 + Mean135)/4, 7, Mean000[7])
FMin  = np.insert(np.min([np.delete(Min000,7),Min090,Min045,Min135],axis=0), 7, Min000[7])
FMax  = np.insert(np.max([np.delete(Max000,7),Max090,Max045,Max135],axis=0), 7, Max000[7])

fig, axs = plt.subplots(5, 1, sharex=True, sharey=True, figsize=(12,15))
fig.subplots_adjust(hspace=0.01)
fig.suptitle('FWHM range vs. off-axis angles over every azimuthal axis',fontsize=18,y=0.92)
## 00
axs[0].plot(np.sort(XX_all[0:15]), Mean000,
            'o',c='firebrick',label='mean at 0$^\circ$')
axs[0].plot(np.sort(XX_all[0:15]), Min000,
            '+',c='firebrick',label='min & max')
axs[0].plot(np.sort(XX_all[0:15]), Max000,
            '+',c='firebrick')
axs[0].fill_between(np.sort(XX_all[0:15]), Min000, Max000, alpha=0.2,color='firebrick',linewidth=0)
axs[0].set_ylim(0,15)
axs[0].set_xlabel('Off-axis angle [arcmin]',fontsize=18)
axs[0].set_ylabel('FWHM [arcsec]',fontsize=14)
axs[0].legend(loc=1)
# 90
axs[1].plot(np.sort(YY_all[15:29]), Mean090,
            'o',c='green',label='mean at 90$^\circ$')
axs[1].plot(np.sort(YY_all[15:29]), Min090,
            '+',c='green',label='min & max')
axs[1].plot(np.sort(YY_all[15:29]), Max090,
            '+',c='green')
axs[1].fill_between(np.sort(YY_all[15:29]), Min090, Max090, alpha=0.2,color='green',linewidth=0)
axs[1].set_ylim(0,15)
axs[1].set_xlabel('Off-axis angle [arcmin]',fontsize=18)
axs[1].set_ylabel('FWHM [arcsec]',fontsize=14)
axs[1].legend(loc=1)
# 45
axs[2].plot(np.sqrt(2)*np.sort(XX_all[29:43]), Mean045,
            'o',c='darkblue',label='mean at 45$^\circ$')
axs[2].plot(np.sqrt(2)*np.sort(XX_all[29:43]), Min090,
            '+',c='darkblue',label='min & max')
axs[2].plot(np.sqrt(2)*np.sort(XX_all[29:43]), Max045,
            '+',c='darkblue')
axs[2].fill_between(np.sqrt(2)*np.sort(XX_all[29:43]), Min090, Max045, alpha=0.2,color='darkblue',linewidth=0)
axs[2].set_ylim(0,15)
axs[2].set_xlabel('Off-axis angle [arcmin]', fontsize=18)
axs[2].set_ylabel('FWHM [arcsec]', fontsize=14)
axs[2].legend(loc=1)
# 135
axs[3].plot(np.sqrt(2)*np.sort(XX_all[43:57]), Mean135,
            'o',c='orange',label='mean at 135$^\circ$')
axs[3].plot(np.sqrt(2)*np.sort(XX_all[43:57]), Min135,
            '+',c='orange',label='min & max')
axs[3].plot(np.sqrt(2)*np.sort(XX_all[43:57]), Max135,
            '+',c='orange')
axs[3].fill_between(np.sqrt(2)*np.sort(XX_all[43:57]),Min135, Max135, alpha=0.2,color='orange',linewidth=0)
axs[2].set_ylim(0,15)
axs[3].set_xlabel('Off-axis angle [arcmin]', fontsize=18)
axs[3].set_ylabel('FWHM [arcsec]', fontsize=14)
axs[3].set_xlim(-10,10)
axs[3].legend(loc=1)
# Average
axs[4].plot(np.sort(XX_all[0:15]), FMean,
            'o',c='gray',label='mean of All')
axs[4].plot(np.sort(XX_all[0:15]), FMin,
            '+',c='gray',label='min & max')
axs[4].plot(np.sort(XX_all[0:15]), FMax,
            '+',c='gray')
axs[4].fill_between(np.sort(XX_all[0:15]), FMin, FMax, alpha=0.2,color='gray',linewidth=0)
axs[4].set_ylim(0,15)
axs[4].set_xlabel('Off-axis angle [arcmin]',fontsize=18)
axs[4].set_ylabel('FWHM [arcsec]',fontsize=14)
axs[4].set_xlim(-10,10)
axs[4].legend(loc=1)
plt.savefig(SaveFolder+'FWHM_MinMax_Angle.pdf')
plt.close(fig)

print('Final!')
# For debugging purposes:
pdb.set_trace()
