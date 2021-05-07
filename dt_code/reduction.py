import os
import numpy as np
from tqdm import tqdm
import ccdproc as ccdp
from math import log10
from astropy import units
from astropy.io import fits
from astropy.time import Time
from astropy.table import Table
from astropy.nddata import CCDData
from astroquery.simbad import Simbad
from scipy.interpolate import interp1d
from astropy.coordinates import SkyCoord
from scipy.ndimage.filters import percentile_filter

from .observatories import *
from .instruments import *

import warnings
warnings.filterwarnings("ignore")

__all__ = ['SpectraReduction']

class SpectraReduction(object):
    """
    Reducing spectra via box extraction for different
    telescope + instrument setups.
    """

    def __init__(self, fn_dir, ref_file, remove_cr=True,
                 observatory='keck', instrument='graces',
                 reload_raw=False, reload_reduced=False):
        """

        Parameters
        ----------
        fn_dir : path
           The directory where all raw data files are stored.
        ref_file : path
           Reference FITS file for grabbing some header information.
        observatory : str
        instrument : str
        reload_raw : bool, optional
           Reloads raw 2D CCD images already saved in .npy files.
           Default is False.
        reload_reduced : bool, optional
           Reloads reduced spectra. Default is False

        Attributes
        ----------
        fn_dir : path
        science_frames : np.ndarray
        med_dark : np.ndarray
        med_flat : np.ndarray
        """

        self.fn_dir = fn_dir

        self.instrument = instrument
        self.observatory = observatory

        self.instrument_specifics(instrument, ref_file)
        self.observatory_specifics(observatory)

        self.errors = None
        self.interp_wavelengths = None
        self.interp_spectra = None
        self.interp_orders = None
        self.interp_errors = None

        if reload_reduced == True:
            self.reload_spectra()
            self.reload_misc()
            self.science_frames=None
            self.npy_files=None
            self.med_flat=None
            self.med_dark=None

        elif reload_raw == True:
            self.npy_files = np.sort([os.path.join(self.fn_dir, i) for i in
                                      os.listdir(self.fn_dir) if i.endswith('.npy')])
            self.reload_master_frames()
            self.reload_misc()
            self.science_frames = self.load_science_frames()

        else:
            self.resave(remove_cr)
            self.npy_files = np.sort([os.path.join(fn_dir, i) for i in 
                                      os.listdir(fn_dir) if i.endswith('.npy')])

            self.med_dark = self.create_master_files(np.sort([i for i in self.npy_files if
                                                              i.endswith('_BIAS.npy')]))
            np.save(os.path.join(self.fn_dir, 'master_dark.npy'), self.med_dark)

            self.med_flat = self.create_master_files(np.sort([i for i in self.npy_files if
                                                              i.endswith('_FLAT.npy')]))
            np.save(os.path.join(self.fn_dir, 'master_flat.npy'), self.med_flat)

            self.reload_misc()
            self.science_frames = self.load_science_frames()


    def resave(self, remove_cr):
        """
        Resaves FITS files into .npy files
        and extracts observation times. Cosmic rays are
        removed from science frames.

        Attributes
        ----------
        times : astropy.time.Time object
           An array of observation times (in JD).
        npy_files : np.ndarray
           An array of the names of the resaved npy files.
        """
        files = np.sort([os.path.join(self.fn_dir, i)
                         for i in os.listdir(self.fn_dir)
                         if i.endswith('.fits')])

        if len(files) > 0:
            
            npy_files = np.array([], dtype='U100')
            times = np.array([])
            exptimes = np.array([])
            
            for fn in tqdm(files):
                name = fn.split('.')[0]
                
                hdu = fits.open(fn)
                dattype = hdu[0].header[self.dattype_key]

                newname = '{0}_{1}.npy'.format(name, dattype)
                npy_files = np.append(npy_files, newname)

                if dattype == 'OBJECT':
                    times = np.append(times, hdu[0].header[self.time_key])
                    exptimes = np.append(exptimes, hdu[0].header[self.exptime_key])

                    ## Removes cosmic rays
                    if remove_cr == True:
                        ccd = CCDData(hdu[0].data, unit='electron')
                        ccd_removed = ccdp.cosmicray_lacosmic(ccd,
                                                              sigclip=6.5,
                                                              sigfrac=0.3)
                        np.save(newname, ccd_removed.data+0.0)
                    else:
                        np.save(newname, hdu[0].data)
                    #np.save(newname, hdu[0].data)

                else:
                    np.save(newname, hdu[0].data)

            self.times = Time(times, format=self.time_fmt)
            self.exptimes = exptimes
            np.save(os.path.join(self.fn_dir, 'times.npy'), self.times)
            np.save(os.path.join(self.fn_dir, 'exptimes.npy'), self.exptimes)

            self.npy_files = npy_files
            return
        else:
            return("No FITS files found in this directory.")
        
        
    def reload_misc(self):
        """ Reloads miscellaneous other files. """
        self.times = Time(np.load(os.path.join(self.fn_dir, 'times.npy'), allow_pickle=True))
        self.exptimes = np.load(os.path.join(self.fn_dir, 'exptimes.npy'), allow_pickle=True)
        self.barycorr = np.load(os.path.join(self.fn_dir, 'barycorr.npy'), allow_pickle=True)
        self.barycorr = self.barycorr * units.km / units.s

    def reload_master_frames(self):
        """
        Reloads pre-saved master dark and flat files.

        Attributes
        ----------
        med_dark : np.ndarray
        med_flat : np.ndarray
        """
        self.med_dark = np.load(os.path.join(self.fn_dir,
                                             'master_dark.npy'),
                                allow_pickle=True)
        self.med_flat = np.load(os.path.join(self.fn_dir,
                                             'master_flat.npy'),
                                allow_pickle=True)
        return

    def create_master_files(self, files):
        """
        Creates master dark/bias/flat files
        to be used for data reduction.

        Parameters
        ----------
        files : np.ndarray
           Takes an array of filenames to create a master 
           frame from.
        
        Returns
        ----------
        med_arr : np.ndarray
           Median frame from observations.
        """

        med_arr = []

        for fn in files:
            d = np.load(fn, allow_pickle=True)
            med_arr.append(d)
        med_arr = np.nanmedian(np.array(med_arr), axis=0)

        return med_arr
        

    def load_science_frames(self):
        """
        Creates master array of science observations that
        are bias subtracted and flat removed.

        Returns
        ----------
        science_frames : np.ndarray
        """
        
        subfiles = np.sort([i for i in self.npy_files if 
                            i.endswith('_OBJECT.npy')])

        for i in range(len(subfiles)):
            dat = np.load(subfiles[i])
            if i == 0:
                science_frames = np.zeros((len(subfiles),
                                           dat.shape[0],
                                           dat.shape[1]))
            science_frames[i] = (dat - self.med_dark) / self.med_flat

        return science_frames


    def barycentric_correction(self, target):#, observatory='keck'):
        """
        Calculates barycentric correction

        Parameters
        ----------
        target : str
           Name of the source being observed. (RA, Dec) of the target
           will be found using astroquery. 
        observatory : str
           The name of the observatory used in barycentric correction.
           Default is 'keck'.

        Attributes
        ----------
        barycorr : np.ndarray
           Barycentric corrections per observation time for a given 
           observatory. Correction given in units of km / s.
        """        
        results = Simbad.query_object(target)
        self.ra = results['RA'][0]
        self.dec = results['DEC'][0]

        barycorr = np.zeros(len(self.times))

        coords = SkyCoord(':'.join(i for i in self.ra.split(' ')), 
                          ':'.join(i for i in self.dec.split(' ')),
                         unit=(units.hourangle, units.deg)) 

        for i in range(len(self.times)):
            barycorr[i] = coords.radial_velocity_correction('heliocentric',
                                                            obstime=self.times[i],
                                                            location=self.geodetic).to(units.km/units.s).value
        self.barycorr = barycorr * units.km / units.s
        np.save(os.path.join(self.fn_dir, 'barycorr.npy'), self.barycorr.value)
        return


    def instrument_specifics(self, instrument, ref_file):
        """
        Uses wavelength and order solution provided
        by an instrument's analysis pipeline.

        Parameters
        ----------
        ref_file : str
           The path + name of the reference file.
        instrument : str, optional
           The name of the instrument. Default is 'GRACES'.

        Attributes
        ----------
        wavelength_ref : np.ndarray
           Array of wavelengths to use as a reference.
        orders_ref : np.ndarray
           Array of order numbers to use as a reference.
        order_start : int
           The starting order for wavelength extraction.
        discrete_model : np.ndarray
           The discrete values to extract orders.
        """
        inst = Instruments(instrument, ref_file)

        keys = list(inst.__dict__.keys())
        for key in keys:
            setattr(self, key, getattr(inst, key))
        return


    def observatory_specifics(self, observatory):
        """
        Gets attributes specific to the observatory.
        Used for barycentric correction of the wavelengths.

        Attributes
        ----------
        tel_lat : float
           Latitude location of the telescope.
        tel_lon : float
           Longitude location of the telescope.
        tel_height : float
           Height of the telescoope.
        geodectic : astropy.EarthLocation
           Earth Location of the telescope.
        """
        obs = Observatories(observatory)
         
        keys = list(obs.__dict__.keys())
        for key in keys:
            setattr(self, key, getattr(obs, key))
        return

    def grid_wavelength(self, wavelength, spectra, length):
        """
        Interpolates wavelength and spectra onto a finer grid.

        Returns
        -------
        wavelength : np.ndarray
        spectra : np.ndarray
        """
        interp_waves = np.zeros(length)
        interp_spect = np.zeros(length)
        
        start = wavelength[0] + 0.0
        end = wavelength[-1] + 0.0
        
        redstart = np.nanmax(start)
        blueend  = np.nanmin(end)
        
        finer_wavelength = np.logspace(log10(redstart), log10(blueend),
                                       length, base=10.0)
        
        f = interp1d(wavelength, spectra)
        
        return finer_wavelength, f(finer_wavelength)

    
    def extract_data(self, cutends=350, percentile=95, size=150, deg=8,
                     interpolate=True, interp_factor=3, err_dir=None):
        """
        Extracts the spectra per order in each file. 

        Parameters
        ----------
        cutends : int, optional
           The number of indices to cut from the both
           ends of each order. Default is 350.
        interp_factor : int, optional
           The factor to interpolate the wavelength
           and spectrum grid over. Default is 3.

        Attributes
        ----------
        wavelengths : np.ndarray
        spectra : np.ndarray
        corrected_spectra : np.ndarray
           Blaze corrected spectra.
        orders : np.ndarray
        """
        c = 2.998 * 10**8 * units.m / units.s # speed of light

        fluxes = np.zeros((len(self.times),
                           self.discrete_model.shape[0]-1,
                           self.discrete_model.shape[1]))

        for i in tqdm(range(len(self.times))):

            dopshift = ((self.wavelength_ref * self.barycorr[i]) / c).to(units.nm)
            wavelength = (self.wavelength_ref - dopshift).to(units.nm).value

            data = self.science_frames[i] + 0.0

            for j in range(self.discrete_model.shape[0]-1):
                order = self.order_start + j

                top = self.discrete_model[j] + 0
                avg_height = np.nanmedian(np.abs(self.discrete_model[j+1] -
                                                 self.discrete_model[j]))
                bottom = np.array(top + avg_height, dtype=int)

                for k in range(top.shape[0]):
                    fluxes[i][j][k] = np.nansum(data[k, top[k]:bottom[k]])

                if i == 0 and j == 0:
                    wavelengths = np.zeros((len(self.times),
                                            self.discrete_model.shape[0]-1,
                                            len(fluxes[i][j])))
                    orders = np.zeros((len(self.times),
                                       self.discrete_model.shape[0]-1,
                                       len(fluxes[i][j])-2*cutends))

                    corrected_wavelengths = np.zeros((len(self.times),
                                               self.discrete_model.shape[0]-1,
                                               len(fluxes[i][j])-2*cutends))
                    corrected_flux = np.zeros((len(self.times),
                                               self.discrete_model.shape[0]-1,
                                               len(fluxes[i][j])-2*cutends))

                # Interpolating wavelengths
                wave = self.wavelength_ref[self.orders_ref==order].value
                newwaves = np.logspace(log10(wave[0]), log10(wave[-1]),
                                       len(fluxes[i][j]), base=10.0)

                # Fit and remove blaze function
                filt = percentile_filter(fluxes[i][j][cutends:-cutends],
                                         percentile=percentile,
                                         size=size)
                fit = np.polyfit(newwaves[cutends:-cutends], filt, deg=deg)
                model = np.poly1d(fit)
                

                corrected_flux[i][j] = (fluxes[i][j][cutends:-cutends] / 
                                        model(newwaves[cutends:-cutends]))
                wavelengths[i][j] = newwaves
                corrected_wavelengths[i][j] = newwaves[cutends:-cutends]
                orders[i][j] = np.full(len(newwaves[cutends:-cutends]), order)

                if interpolate:
                    if i == 0 and j == 0:
                        interp_waves = np.zeros( (corrected_wavelengths.shape[0], 
                                                  corrected_wavelengths.shape[1],
                                                  corrected_wavelengths.shape[2]*interp_factor) )
                        interp_spect = np.zeros( interp_waves.shape )
                        interp_ordrs = np.zeros( interp_waves.shape )

                    iw, iss = self.grid_wavelength(corrected_wavelengths[i][j],
                                                   corrected_flux[i][j],
                                                   interp_spect.shape[2])
                    interp_waves[i][j] = iw
                    interp_spect[i][j] = iss
                    interp_ordrs[i][j] = np.full(interp_spect.shape[2], order)

        if interpolate:
            self.interp_wavelengths = interp_waves + 0.0
            self.interp_spectra = interp_spect + 0.0
            self.interp_orders = interp_ordrs + 0.0
            
        self.wavelengths = wavelengths + 0.0
        self.spectra = fluxes + 0.0
        self.corrected_spectra = corrected_flux + 0.0
        self.corrected_wavelengths = corrected_wavelengths + 0.0
        self.orders = orders + 0

        if err_dir is not None:
            self.error_extraction(err_dir, cutends, interpolate)
        

    def error_extraction(self, directory, cutends, interpolate):
        """
        Gets associated errors from previously reduced data

        Attributes
        ----------
        corrected_errors : np.ndarray
        """
        files = np.sort([os.path.join(directory, i) for i in os.listdir(directory)])

        err = np.zeros(self.corrected_spectra.shape)
        if interpolate:
            interp_err = np.zeros(self.interp_wavelengths.shape)


        for i in range(len(files)):
            hdu = fits.open(files[i])
            
            for j in range(self.corrected_spectra.shape[1]):
                order = self.order_start + j
                q = np.where( (hdu[0].data[0] == order) &
                              (hdu[0].data[4] >= self.corrected_wavelengths[i][j][0]-0.1) &
                              (hdu[0].data[4] <= self.corrected_wavelengths[i][j][-1]+0.1) )[0]
                
                f = interp1d(hdu[0].data[4][q], np.sqrt(hdu[0].data[11][q]))
                err[i][j] = f(self.corrected_wavelengths[i][j])

                if interpolate:
                    interp_err[i][j] = f(self.interp_wavelengths[i][j])
                              
            hdu.close()

        self.errors = err + 0.0
        if interpolate:
            self.interp_errors = interp_err + 0.0


    def save_spectra(self, fn_dir=None):
        """
        Exports spectra as .npy files.

        Parameters
        ----------
        fn_dir : str, optional
           Directory where to save the reduced files to.
           Default is the initialized fn_dir.
        """
        if fn_dir is None:
            fn_dir = self.fn_dir

        np.save(os.path.join(fn_dir, 'raw_wavelengths.npy'),
                self.wavelengths)
        np.save(os.path.join(fn_dir, 'raw_spectra.npy'),
                self.spectra)

        np.save(os.path.join(fn_dir, 'corrected_wavelengths.npy'),
                self.corrected_wavelengths)
        np.save(os.path.join(fn_dir, 'corrected_spectra.npy'),
                self.corrected_spectra)
        np.save(os.path.join(fn_dir, 'corrected_orders.npy'),
                self.orders)

        if self.errors is not None:
            np.save(os.path.join(fn_dir, 'corrected_errors.npy'),
                    self.errors)

        if self.interp_wavelengths is not None:
            
            np.save(os.path.join(fn_dir, 'interpolated_wavelengths.npy'),
                    self.interp_wavelengths)
            np.save(os.path.join(fn_dir, 'interpolated_spectra.npy'),
                    self.interp_spectra)
            np.save(os.path.join(fn_dir, 'interpolated_orders.npy'),
                    self.interp_orders)

            if self.interp_errors is not None:
                np.save(os.path.join(fn_dir, 'interpolated_errors.npy'),
                        self.interp_errors)
        return
            
    def reload_spectra(self):
        """
        Reloads already saved 1D spectra.
        """
        fn_dir = self.fn_dir

        self.wavelengths = np.load(os.path.join(fn_dir, 'raw_wavelengths.npy'), allow_pickle=True)
        self.spectra = np.load(os.path.join(fn_dir, 'raw_spectra.npy'), allow_pickle=True)
        
        self.corrected_wavelengths = np.load(os.path.join(fn_dir, 'corrected_wavelengths.npy'),
                                             allow_pickle=True)
        self.corrected_spectra = np.load(os.path.join(fn_dir, 'corrected_spectra.npy'),
                                             allow_pickle=True)
        self.orders = np.load(os.path.join(fn_dir, 'corrected_orders.npy'), allow_pickle=True)

        try:
            self.errors = np.load(os.path.join(fn_dir, 'corrected_errors.npy'), allow_pickle=True)
        except:
            print('No error file found.')
            self.errors = None

        
        try:
            self.interp_wavelengths = np.load(os.path.join(fn_dir, 'interpolated_wavelengths.npy'),
                                              allow_pickle=True)
            self.interp_spectra = np.load(os.path.join(fn_dir, 'interpolated_spectra.npy'),
                                          allow_pickle=True)
            self.interp_orders = np.load(os.path.join(fn_dir, 'interpolated_orders.npy'),
                                         allow_pickle=True)
        except:
            print('No interpolated spectra files found.')
            self.interp_wavelengths=None
            self.interp_spectra=None
            self.interp_orders=None

        try:
            self.interp_errors = np.load(os.path.join(fn_dir, 'interpolated_errors.npy'),
                                         allow_pickle=True)
        except:
            print('No interpolated error file found.')
            self.interp_errors=None

        return
