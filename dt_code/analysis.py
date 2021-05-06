import numpy as np
from astropy import units
from matplotlib import gridspec
import matplotlib.pyplot as plt

from .utils import *

__all__ = ['DTAnalysis']

class DTAnalysis(object):
    """
    Makes all the pretty waterfall plots and stuff.
    """

    def __init__(self, wavelengths, spectra, errors, orders, obstimes):
        """
        
        Parameters
        ----------
        wavelength : np.ndarray
        spectra : np.ndarray
        error : np.ndarray
        obstime : np.ndarray
        """
        self.wavelengths = wavelength
        self.spectra = spectra
        self.errors = errors
        self.orders = orders
        self.obstimes = obstimes


    def expanding_bins(self, lines, repeat=3, subtract=True, template=None):
        """
        Increases bin width for the waterfall plots by a factor of `repeat`.

        Parameters
        ----------
        lines : np.ndarray
           Spectral lines
        repeat : int, optional
           Factor by which to increase the binsize. Default is 3.
        subtract : bool, optional
           Whether to plot the residuals or not. Default is True.
        template : np.ndarray, optional
           Template to subtract from the lines. Default is a median across
           all observations.

        Returns
        -------
        binned : np.ndarray
           New binned waterfall plot.
        template : np.ndarray
           Template subtracted off (if subctract=True).
        """
        if template is None:
            template = np.nanmedian(lines, axis=0)
        binned = np.zeros( (len(self.obstimes)*repeat, len(lines[0]))  )
            
        z = 0
        for i in range(len(lines)):
            if subtract == True:
                binned[z:z+repeat] = lines[i] - template
            else:
                binned[z:z+repeat] = lines[i]
            z += repeat

        return binned, template

    def to_velocity(self, wave, flux=None, line=None):
        """
        Converts wavelength --> velocity (km / s). If no
        center reference point is passed in, the middle of
        the array == 0 km/s.

        Parameters
        ----------
        wave : np.ndarray
           Wavelength array to convert.
        flux : np.ndarray, optional
           Spectrum array. If passed in, will used the 
           minimum in the spectrum as 0 km/s point. Default
           is None.
        line : float, optional
           Wavelength to set as 0 km/s point. Default is None.

        Returns
        -------
        rv_km_s : np.ndarray
           Array of velocities in km/s.
        argmid : int
           Index of 0 km/s in wavelength space.
        """
        if line != None:
            argmid = np.where(wave>=line)[0][0]
        else:
            argmid = int(len(wave)/2)
            if flux is not None:
                argmid = np.argmin(flux)

        lambda0 = wave[argmid] + 0.0
        rv_m_s = ((wave - lambda0)/lambda0 * 3e8)*units.m/units.s
        rv_km_s = rv_m_s.to(units.km/units.s)
        return rv_km_s, argmid


    def normalizing_depth(self, reg=None):
        """
        Normalizes the line based on the area.

        Parameters
        ----------
        reg : np.ndarray
           2D array of lower and upper wavelengths
           to search over. Default is None.

        Returns
        -------
        norm_spect : np.ndarray
        """
        if reg is None:
            raise valueError('Need to pass in a wavelength region to normalize over.')
        
        normalized = np.zeros(self.spectra.shape)
        
        for i,flux in enumerate(self.spectra):
        
            region = ((self.wavelengths[i]>=reg[0]) & 
                      (self.wavelengths[i]<=reg[1]))
            
            norm_around_zero = flux - np.nanmedian(flux[~region])

            area = np.trapz(norm_around_zero[region], 
                            self.wavelengths[i][region])
            normalized[i] = norm_around_zero/np.abs(area)
            
        return normalized
