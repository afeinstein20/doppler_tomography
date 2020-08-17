import os
import numpy as np
from astropy.time import Time
from astropy import units as u
from astropy import constants as c
from scipy.interpolate import interp1d
from astropy.coordinates import SkyCoord, EarthLocation

__all__ = ['RVPipeline']

class RVPipeline(object):
    """
    Takes spectra and returns RVS.
    """

    def __init__(self, wavelength, spectrum, orders):
        self.w = wavelength
        self.s = spectrum
        self.o = orders
        
        self.telluric_wavelength = None
        self.telluric_spectrum   = None

        self.w_masked = None
        self.s_masked = None
        self.o_masked = None

    def load_tellurics(self, directory=None):
        """
        Loads in telluric .hdf5 file

        Parameters
        ----------
        directory : str, optional

        Attributes
        ----------
        telluric_wavelength : np.ndarray
        telluric_spectrum : np.ndarray
        """
        import h5py

        if directory == None:
            directory = '.'
        
        files = os.listdir(directory)
        files = np.sort([os.path.join(directory,i) for i in files if i.endswith('.hdf')])

        wave_model, spec_model = np.array([]), np.array([])

        for fn in files:
            f = h5py.File(os.path.join(directory, fn), 'r')
            for i, key in enumerate(f['wavelength_solution']['fiber_2'].keys()):                
                wave_model = np.append(wave_model, 
                                       f['wavelength_solution']['fiber_2'][key][()])
                spec_model = np.append(spec_model,
                                       f['telluric_model']['fiber_2'][key][()])

        
        wave_model, spec_model = zip(*sorted(zip(wave_model, spec_model)))

        self.telluric_wavelength = np.array(wave_model)
        self.telluric_spectrum   = np.array(spec_model)

        return
    

    def mask_tellurics(self, cutends=3000, telluric_limit=0.992, offset=0.26,
                       percentage=0.65):
        """
        Masks telluric features in spectra.
        
        Parameters
        ----------
        cutends : int, optional
             The number of data points to cut from both ends of the order.
        telluric_limit : float, optional
             The minimum strength of the telluric features to mask.
        offset : float, optional
             Wavelength offset between telluric model and data.
        percentage : float, optional
             Upper limit of allowable percentage of tellurics in a given order.

        """
        if type(self.telluric_wavelength) != np.ndarray:
            return("No telluric model found.")

        cleaned_w = np.zeros(len(self.w), dtype=np.ndarray)
        cleaned_s = np.zeros(len(self.s), dtype=np.ndarray)
        cleaned_o = np.zeros(len(self.o), dtype=np.ndarray)

        for i in range(len(self.w)):
            tw, ts, to = np.array([]), np.array([]), np.array([])

            for uo in np.unique(self.o[i]):
                q = self.o[i] == uo

                if cutends > 0:
                    shortw = self.w[i][q][cutends:-cutends]
                    shorts = self.s[i][q][cutends:-cutends]
                    shorto = self.o[i][q][cutends:-cutends]
                else:
                    shortw = self.w[i][q]
                    shorts = self.s[i][q]
                    shorto = self.o[i][q]

                try:
                    inds = np.where( (self.telluric_wavelength >= np.nanmin(shortw)-1.0) &
                                     (self.telluric_wavelength <= np.nanmax(shortw)+1.0) )[0]
                    
                    # Mapds tellurics onto the same grid as the data
                    interp = interp1d(self.telluric_wavelength[inds]-offset,
                                      self.telluric_spectrum[inds])
                    mapped = interp(shortw)
                    mask   = np.where(mapped > telluric_limit)[0]
                    
                    if len(mask)/len(mapped) > percentage:
                        tw = np.append(tw, shortw[mask])
                        ts = np.append(ts, shorts[mask])
                        to = np.append(to, shorto[mask])

                except:
                    pass

            cleaned_w[i] = tw
            cleaned_s[i] = ts
            cleaned_o[i] = to

        self.equalize_orders(w=cleaned_w,
                             s=cleaned_s,
                             o=cleaned_o)

        return

    def equalize_orders(self, w=None, s=None, o=None):
        """
        Sets each order in each spectra to be the same length for nice
        cross-correlation capabilities.

        Attributes          
        ----------     
        w_masked : np.ndarray
        s_masked : np.ndarray
        o_masked : np.ndarray 
        """
        lengths = np.zeros(len(np.unique(o[0])), dtype=int)

        for j, uo in enumerate(np.unique(o[0])):
            temp_len = np.zeros(len(w))
            for i in range(len(w)):
                temp_len[i] = len(np.where(o[i] == uo)[0])
            lengths[j] = np.nanmin(temp_len)

        equal_wave = np.zeros( (len(w), np.nansum(lengths)), dtype=np.ndarray )
        equal_spec = np.zeros( (len(w), np.nansum(lengths)), dtype=np.ndarray )
        equal_ordr = np.zeros( (len(w), np.nansum(lengths)), dtype=np.ndarray )

        for j, uo in enumerate(np.unique(o[0])):

            if j == 0:
                inds = [0, lengths[j]]
            else:
                inds = [np.nansum(lengths[:j]), np.nansum(lengths[:j+1])]

            # Go through each order and remove points to
            # get each spectra to the same shape post telluric masking
            for i in range(len(w)):
                q = o[i] == uo
            
                diff = np.abs(len(w[i][q]) - lengths[j])

                if diff > 0:
                    if diff == 1:
                        cw = w[i][q][1:]
                        cs = s[i][q][1:]
                        co = o[i][q][1:]
                    else:
                        if diff % 2 == 0:
                            cw = w[i][q][int(diff/2):-int(diff/2)]
                            cs = s[i][q][int(diff/2):-int(diff/2)]
                            co = o[i][q][int(diff/2):-int(diff/2)]
                        else:
                            cw = w[i][q][int(diff/2):-int(diff/2)-1]
                            cs = s[i][q][int(diff/2):-int(diff/2)-1]
                            co = o[i][q][int(diff/2):-int(diff/2)-1]
                else:
                    cw = w[i][q]
                    cs = s[i][q]
                    co = o[i][q]

                equal_wave[i][inds[0]:inds[1]] = cw
                equal_spec[i][inds[0]:inds[1]] = cs
                equal_ordr[i][inds[0]:inds[1]] = co

        self.w_masked = equal_wave
        self.s_masked = equal_spec
        self.o_masked = equal_ordr

        return 
        
        
    def xcorrelate(self, mode='same', percentile=25):
        """
        Cross-correlates each order with a template to extract RVs.

        Parameters
        ----------
        mode : str, optional
             Mode to use in np.correlate.
        percentile : float, optional
             Percentile value to make the template from.
        
        Attributes
        ----------
        rvs : np.ndarray
        """
        
        if type(self.w_masked) != np.ndarray:
            return("Need to remove tellurics and clean spectra first.")

        orders = np.unique(self.o_masked[0])

        rvs = np.zeros( (len(orders), len(self.w_masked)) )

        for j, uo in enumerate(orders):
            peaks = np.zeros(len(self.w_masked))

            for i in range(len(self.w_masked)):
                q = self.o_masked[0] == uo
                
                if i == 0:
                    template = np.percentile(self.s_masked, percentile, axis=0)[q]
                    subwave = self.w_masked[i][q] + 0.0

                corr = np.correlate(template - np.median(template),
                                    self.s_masked[i][q] - np.median(self.s_masked[i][q]),
                                    mode=mode)
                c_x = np.arange(np.argmax(corr)-4, np.argmax(corr)+5, 1, dtype=int)
                print(corr, c_x)
                fit = np.polyfit(c_x-np.argmax(corr), corr[c_x], deg=2)
                maximum = -fit[1]/(2*fit[0])
                peaks[i] = maximum

            
            rv = np.nanmedian(np.diff(np.log(subwave))) * 3e8 * peaks
            rvs[j] = rv
            
        self.rvs = rvs


