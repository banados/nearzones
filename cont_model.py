#!/usr/bin/env python
from __future__ import print_function, division
import argparse
import os

import numpy as np
from scipy import optimize

import matplotlib.pyplot as plt
from astropy.io import ascii
import astropy.units as u

from linetools.spectra.io import readspec

from array_utils import between, get_ind


EXAMPLES = '''

cont_model.py qso_info.txt 


 '''
 

description = '''
Input a spectrum and a redshift
 
 Fit the continuum in regions typically free of emission lines.
 This is meant for objects with no optimal spectra or NIR spectra so that the 
 continuum fit is not reliable. 

The continuum is not very sophisticated, it consists of a power law continuum with a fixed power index.

f_lambda = Constant X wavelength ** alpha_lambda

Plus two Gaussians fitted to Lya and NV. This is for cases where you are studying regions at rest-frame 
wavelength less than 1300 Angstroms.

The power index is defined by the user but typical values are:

 alpha_lambda = -1.5 (VandenBerk 2001)
 alpha_lambda = -1.7 (Selsing 2016)
 
We subtract the "fitted" continuum (in practice is just a vertical scale). And then 
we calculate the EW of Lya+Nv between lambda=1160 - 1290 (following Diamond-Stanic 2009)

 The outputs are a spectrum file with the continuum model and a figure showing the spectrum and the model.
                '''

#INCLUDE is used in my code to include only line-free regions in the fitting
include_min = [1285, 1315, 1340, 1425, 1680, 1975, 2150]
include_max = [1295, 1325, 1375, 1470, 1710, 2050, 2250]

#power law definition
powerlaw = lambda x, amp, index: amp * (x ** index)

lya0 =  1215.67 * u.Angstrom #Angstroms VandenBerk 2001 lab
nv0 = 1240.14 * u.Angstrom #Angstrom VandenBerk 2001 lab


def final_plot(ax):
    ax.plot(obs_spc.wavelength, continuum_model, lw=2, color="gray")

    ax.set_xlabel('$\mathrm{Wavelength\\ (\AA)}$')
    ax.set_ylabel(r'$f_\lambda \; \mathrm{(erg\; s^{-1}\;cm^{-2}\;\AA^{-1}})$')
    ax.set_xlim(lya0.value * (redshift+0.4), lya0.value * (redshift+1.8))
    ax.set_ylim(bottom=-0.1e-17)
    plt.title(output)
    f.savefig(output + ".png")
    print(output + ".png", " created")
    plt.close()


def get_free_line_region(spc, zp1):
    '''
    Get slices with free line regions
    '''
    
    wave = spc.wavelength.value
    flux = spc.flux.value
    if spc.sig_is_set:
        error = spc.sig.value
    else:
        error = None
    #get minimum and maximum indices from the regions that are also in spectrum
    
    imin = get_ind(val=wave[0], arr=include_min)
    imax = get_ind(wave[-1], include_max) + 1
    incl_min = include_min[imin:imax]
    incl_max = include_max[imin:imax]
    
    ia = get_ind(incl_min[0], wave)
    ib = get_ind(incl_max[0], wave)
    
    #initialize the slices that will be used to fit
    w2f = wave[ia:ib]
    f2f = flux[ia:ib]
    if error is None:
        e2f = None
    else:
        e2f = error[ia:ib]
        

    for a,b in zip(incl_min[1:], incl_max[1:]):
        i1 = get_ind(a, wave)
        i2 = get_ind(b, wave)
        w2f = np.hstack((w2f, wave[i1:i2]))
        f2f = np.hstack((f2f, flux[i1:i2]))
        if error is not None:
            e2f = np.hstack((e2f, error[i1:i2])) /zp1
    
    #mask negatives
    mask = f2f > 0     
    w2f = w2f[mask]
    f2f = f2f[mask]
    if error is not None:
        e2f = e2f[mask]

    #wave2fit, flux2fit, error2fit
    return w2f * zp1, f2f/zp1, e2f


def get_gaussian_fits(wave, flux, error, redshift, ax):
    from astropy.modeling import models, fitting

    lyaredshifted = lya0.value * (1.+redshift) 
    nvredshifted = nv0.value * (1.+redshift)
    uplim = 1270. * (1+redshift)
    ax.axvline(lyaredshifted, ls="--")
    ax.axvline(nvredshifted, ls="--")

    m = between(wave, lyaredshifted, uplim)
    w = wave[m]
    f = flux[m]

    #Gaussian for Lya
    g1 = models.Gaussian1D(amplitude=f.max(),
                 mean=lyaredshifted, stddev=19.46 * (1+redshift))
    #Gaussian for NV
    g2 = models.Gaussian1D(amplitude=f.max()*0.2,
                 mean=nvredshifted, stddev=2.71 * (1+redshift))

    gg_init = g1 + g2
    fitter = fitting.LevMarLSQFitter()
    gg_fit = fitter(gg_init, w, f)

    plot_gauss_params(ax, gg_fit)

    return gg_fit(wave)


def get_powerlaw_continuum(data, ax):
    """
    Fit a power law continuum and return the continuum and the same wavelength as the input spectrum
    """
    #read spectrum
    zp1 = 1.0 + data["redshift"]
    spc = read_restframe_spectrum(data)
    w2f, f2f, e2f = get_free_line_region(spc, zp1)

    ax.plot(w2f, f2f, 'b-', lw=2)
    amp, index, ampErr, label_fit =powerlaw_fit(w2f, f2f, e2f, slopecont=data["alpha_lambda"])
    plot_pl_fit(ax, obs_spc.wavelength, amp, index, fit_label=label_fit, color="blue", lpos=1)
    powerlawcont = powerlaw(obs_spc.wavelength.value, amp, index)
    cont_subtracted = obs_spc.flux.value - powerlawcont

    return powerlawcont, cont_subtracted



def plot_pl_fit(ax, wave, amp, index, fit_label, color="green", lpos=1):
    ax.plot(wave, powerlaw(wave, amp, index), label=fit_label,
            color=color, lw=1.5)
    yp=0.8
    if lpos == 1:
        xp =0.7
    elif lpos == 2:
        xp=0.3
    ax.text(xp, yp, fit_label, color=color,
         horizontalalignment='center',
         verticalalignment='center',
         transform = ax.transAxes,
         bbox = {'fc':'white', 'pad':10, 'alpha':0.5})


def plot_gauss_params(ax, gg_fit, color="gray", lpos=1):
    label_fit = r'Gaussians'
    label_fit += "\n"
    label_fit += 'Ampl1: {:.3g} '.format(gg_fit.amplitude_0.value) + r',  mean1:' + '{:.3g} ' .format(gg_fit.mean_0.value)  + r',  std1:' + '{:.3g} \n' .format(gg_fit.stddev_0.value)
    label_fit += 'Ampl2: {:.3g} '.format(gg_fit.amplitude_1.value) + r',  mean2:' + '{:.3g} ' .format(gg_fit.mean_1.value)  + r',  std2:' + '{:.3g} ' .format(gg_fit.stddev_1.value)

    yp=0.5
    if lpos == 1:
        xp =0.67
    elif lpos == 2:
        xp=0.3
    ax.text(xp, yp, label_fit, color=color,
         horizontalalignment='center',
         verticalalignment='center',
         transform = ax.transAxes, fontsize="small",
         bbox = {'fc':'white', 'pad':10, 'alpha':0.5})


def powerlaw_fit(w2f, f2f, e2f, slopecont):
    """
    Based on the example Fitting a power-law to data with errors
    #http://wiki.scipy.org/Cookbook/FittingData
    
    """

    logx = np.log10(w2f)
    logy = np.log10(f2f)
    
    if e2f is not None:
        logyerr = e2f / f2f
    else:
        print("Caution: Error is None")
        logyerr = 1.0

    # define our (line) fitting function
    fitfunc = lambda p, x: p + slopecont * x
    errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err     
    pinit = [1.0]
    out = optimize.leastsq(errfunc, pinit,
                      args=(logx, logy, logyerr), full_output=1)

    pfinal = out[0]
    covar = out[1]

    index = slopecont
    amp = 10.0**pfinal[0]
    #print("amp", amp)
    ampErr = np.sqrt( covar[0][0] ) * amp

    label_fit = r'$f_\lambda = C \times \lambda^\beta$'
    label_fit += "\n"
    label_fit += 'C = {:.3g} '.format(amp) + r'$\pm$' + '{:.3g} \n' .format(ampErr)
    label_fit += r'$\beta$ =' + '{:.3g}'.format(index) 

    return amp, index, ampErr, label_fit


def read_restframe_spectrum(data):
    d = readspec(data['spectra'])
    spc = redshift_spectrum_to_restframe(spc=d, redshift=data["redshift"])
 
        
    return spc


def redshift_spectrum_to_restframe(spc, redshift):
    """
    what the name says
    """
    zp1 = 1.0 + redshift
    spc.wavelength = spc.wavelength / zp1
    spc.flux = spc.flux * zp1 #to conserve the integrated flux
    spc.sig = spc.sig * zp1 

    return spc


def write_spectrum_with_continuum():
    out = output + ".spc"
    obs_spc.write_to_ascii(out)
    print(out, " has been created")


def parse_arguments():
    
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES)
        
    parser.add_argument('info', type=str,
                      help='Fits or text file containing the information required.\
                      The columns needed are: name (just an identification), redshift, spectra (path to spectra), and alpha_lambda (power law index) ')

    return parser.parse_args()
    
if __name__ == '__main__':

    args=parse_arguments()

    
    #read info file
    table = ascii.read(args.info)
    #the output file will be the same but with new columns
    print("Number of entries: ", len(table))

    #start the loop
    for i, row in enumerate(table):
        print("="*60)
        print(row['name'])
        obs_spc = readspec(row["spectra"])

        filename, file_extension = os.path.splitext(row["spectra"])
        output = filename + "_" +"z" + str(row["redshift"]) + str(row["alpha_lambda"]) + "_cont"

        f, ax = plt.subplots()

        ax.plot(obs_spc.wavelength, obs_spc.flux, lw=3, color="k")

        continuum_powerlaw, continuum_subtracted = get_powerlaw_continuum(data=row, ax=ax)
        #ax.plot(obs_spc.wavelength, continuum_subtracted, lw=3, color="k")

        continuum_lya = get_gaussian_fits(wave= obs_spc.wavelength.value, flux=continuum_subtracted,
                                        error=obs_spc.sig, redshift=row["redshift"], ax=ax)

        continuum_model = continuum_powerlaw + continuum_lya
        obs_spc.co = continuum_model

        final_plot(ax)

        write_spectrum_with_continuum()