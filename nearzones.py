#!/usr/bin/env python
from __future__ import print_function, division
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.cosmology import z_at_value

from linetools.spectra.xspectrum1d import XSpectrum1D
from linetools.spectra.io import readspec

from array_utils import between
from cont_model import lya0


EXAMPLES = '''

nearzones.py qso_b1.7.spc qsocont_info.txt 


 '''
 

description = '''
Input a file with the path to quasar spectra with a continuum model included, the 
systemuc redshift is also required. As input can be used the output of 
cont_model.py for example.

1.- Determine the transmission by dividing observed spectrum by the model.
2.- The transmission is smoothed to a resulution of 20 AA.

Note that for now we use Planck15 cosmology, but this can be changed easily upon request.
                '''


def find_nearzone_radius(t):
    """
    Receives an XSpectrum1D object with transmission as flux and 
    pdistlya as Proper distance with respect to Lya.

    This function determines the corresponding radius when the transmission 
    drops to 10%. It also estimates the error as the width of the bin.
    
    """
    #Only consider the relevant part of the array: -1 -- 15 Mpc
    m = between(t.pdistlya.value, -1, 15)
    x = t.pdistlya[m]
    y = t.flux[m]

    my = y < 0.1
    #The last element of the masked rnz array contains the first bin when
    #the transmission falls to 10%
    rnz = x[my][-1]
    rnze = np.mean(np.abs(np.diff(x))) * 0.5

    print("*************")
    print("Near zone: {:.2f} pm {:.2f}".format(rnz, rnze))
    print("*************")

    return rnz, rnze


def nearzone_corrected(nz, M1450):
    """
    Scaled Near Zone to an absolute magnitude of M1450=-27 for comparison
    """
    nzc = nz * 10 ** ((0.4/3.) * (27. + M1450))
    print("Near zone corrected: ", nzc)

    return nzc


def nearzone_plot(t, redshift,
                      output):
    """

    """
    f, ax = plt.subplots()
    #ax.plot(t.wavelength, t.flux, drawstyle="steps-mid", lw=2, color="k")
    ax.plot(t.pdistlya, t.flux, drawstyle="steps-mid", lw=3, color="k")


    #ax.axvline(lya0.value * (1+redshift), ls=":", color="k")
    ax.axvline(0, ls=":", color="k")
    ax.axhline(1.0, ls=":", color="k")
    ax.axhline(1.0, ls=":", color="k")
    ax.axhspan(-0.1, 0.1, color="gray", alpha=0.5)
    ax.set_ylim(-0.01, 1.05)
    ax.set_xlim(12, -1)

    ax.set_xlabel("$\mathrm{Proper\\ distance\\ (Mpc)}$")
    ax.set_ylabel("$\mathrm{Transmission}$")

    plt.title(output)
    rnz, rnze = find_nearzone_radius(t)
    ax.axvline(rnz.value, ls="--", lw=2)
    result_label = "$NZ = {:.2f} \pm {:.2f}$".format(rnz, rnze)

    ax.text(0.3, 0.7, result_label, color="k",
         horizontalalignment='center',
         verticalalignment='center',
         transform = ax.transAxes,
         bbox = {'fc':'white', 'pad':10, 'alpha':0.5})

    f.savefig(output + ".png")
    print(output + ".png", " created")
    plt.close()


def normalize_spectrum(spc):
    """
    The flux is normalized by the continuum
    """
    spc.flux = spc.flux/spc.co
    if spc.sig_is_set:
        spc.sig = spc.sig/spc.co


    

def smooth_specrum(spc, resolution):
    """
    Smooth the spectrum to a resolution in Angstroms.
    It assumes that every wavelength bin is uniformly spaced.
    """
    w = spc.wavelength.to("Angstrom")
    w1 = w[0].value
    npix = np.sum(between(w, w1, w1+20.))
    #print(npix)
    return spc.box_smooth(npix)


def wavelength_to_proper_distance_lya(wave, redshift, cosmo):
    """
    Transform wavelength to proper distance with respect to Lya.
    It requires an XSpectrum1D object, the redshift, and 
    an astropy.cosmology object.

    It will create a new attribute to the spectrum called 
    pdistlya in Mpc
    """
    #Transform wavelength to Lya redshift
    zlya = (wave/lya0).decompose() - 1.
    #Redshift to lookbacktime distance
    lbd = cosmo.lookback_distance(zlya.value)
    #lbd at the systemic redshift of the quasar:
    lbdlya = cosmo.lookback_distance(redshift)
    pdistlya = lbdlya - lbd 

    return pdistlya


def write_transmission(s, output="tmp.tmp"):
    """
    It receives a XSpectrum1D object with transmission as flux and 
    pdistlya argument. It writes a text file with wavelength 
    pdistlya and transmission. This is useful for plotting the 
    results a posteriori.
    """
    out = [s.wavelength, s.pdistlya, s.flux]
    names = ["wave", "pdistlya", "trans"]
    ascii.write(out, output, names=names)
    print(output, " has been created")



def parse_arguments():
    
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES)
        
    parser.add_argument('info', type=str,
                      help='Fits or text file containing the information required.\
                      The columns needed are: name (just an identification), redshift, spectra with continuum model (path to spectra ')


    return parser.parse_args()


if __name__ == '__main__':

    args=parse_arguments()

    #Choose your prefer cosmology here
    from astropy.cosmology import Planck15 as cosmo

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
        output = filename + "_" +"nz" 


        #normalize
        normalize_spectrum(obs_spc)
        #smooth
        smooth_trans = smooth_specrum(obs_spc, resolution=20)

        pdist = wavelength_to_proper_distance_lya(smooth_trans.wavelength,
                                            row["redshift"], cosmo)


        smooth_trans.pdistlya = pdist


        nearzone_plot(smooth_trans, redshift=row["redshift"],
                      output=output)

        write_transmission(smooth_trans, output=output+".txt")
