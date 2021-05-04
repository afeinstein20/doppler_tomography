import numpy as np
from astropy import units
from astropy.io import fits

__all__ = ['Instruments']


class Instruments(object):
    """
    Gets details for specific spectrographs.
    """
    
    def __init__(self, instrument, ref_file):
        """
        Each function returns the same attributes
        specified for different instruments.

        Attributes
        ---------
        ref_file : str
           The reference file to draw wavelength
           and order solutions from.
        wavelength_ref : np.ndarray
           Array of wavelengths to use as a reference.
        orders_ref : np.ndarray
            Array of order numbers to use as a reference.
        order_start : int 
           The starting order for wavelength extraction.
        discrete_model : np.ndarray
           Discrete order models used for spectrum extraction.
        """
        self.ref_file = ref_file

        if instrument.lower() == 'graces':
            self.get_graces()

        else:
            return('Instrument not incorporated yet.')
        return


    def get_graces(self):
        """ Information for GRACES on Gemini North. """

        hdu = fits.open(self.ref_file)
        self.wavelength_ref = hdu[0].data[4] * units.nm
        self.orders_ref = hdu[0].data[0]
        hdu.close()

        self.order_start = 23
        self.discrete_model = np.load('graces_discrete_models.npy')

        self.time_key = 'MJDATE'
        self.exptime_key = 'EXPTIME'
        self.time_fmt = 'mjd'
        self.dattype_key = 'OBSTYPE'

        return
