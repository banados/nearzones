# QSO near zones

Software to estimate quasar near zones.


These are the needed steps:

1. Create a continuum model. We actually just need a model from the region with rest-frame wavelength < 1300 AA. This includes Lya+NV and the continuum blueward of Lya.

    A simple (-istic) way to do this is with the code `cont_model.py`, which requires the redshift and the spectra of the quasar as input. It "fits" a powerlaw continuum with a fix index (chosen by the user). It also fits two Gaussians to Lya and NV based on the input redshift.
    Write a file with wavelength, flux, (error), continuum.

    Example of input file called test.info:

        name   redshift spectra alpha_lambda
        qso1     6.6    qso1.spc  -1.7
        qso2     6.0    qso2.spc  -1.5

    Example:
    ```python
    python cont_model.py test.info
    ```

    Note that you can do the continuum fit in more elaborate ways. Choose the one that makes more sense for your data



2. The code `nearzones.py` receives the spectrum from 1. Normalize the observed spectrum by its continuum. The resultant transmission is smoothed to a resolution of 20 Angstroms (ala Fan et al.). Wavelengths are transformed to Proper distance (Mpc) from rest-frame Lya. It estimates the proper distance when the transmission drops to 10 percent and report the value. A plot is created as a check and a new file with wavelength, proper distance in Mpc wrt Lya, and the transmission is also created.


    Example of input file called testnz.info:

        name   redshift spectra alpha_lambda
        qso1     6.6    qso1_cont.spc  -1.7
        qso2     6.0    qso2_cont.spc  -1.5

    Example:
    ```python
    python nearzones.py testnz.info
    ```
