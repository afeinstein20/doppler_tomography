from astropy import units as u

__all__ =['InstrumentDeets']

class InstrumentDeets(object):
    """
    Retrieves specific instrument keywords.
    """

    def __init__(self, instrument):
        
        if instrument.lower() == 'graces':
            self.graces_keys()


        


    def graces_keys(self):
        """
        Assigns keys for using GRACES.
        """
        self.fwhm = 5.0
        self.hdu_ext = 0
        self.time_key = 'MJDATE'
        self.airmass_key = 'AIRMASS'
        self.wave_key = 4
        self.wave_unit = u.nm
        self.flux_key = 10
        self.order_key = 0
        self.flux_err_key = 11
        self.cut_order_ends = 250
        
