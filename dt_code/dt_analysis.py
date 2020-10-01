import os, sys
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy import units as u

from .utils import *
from .instrument_specifics import *

__all__ = ['DT']


class DT(object):
    """
    Performs doppler tomography analaysis.

    Spectra should be in FITS files.
    """
    
    def __init__(self, path, instrument='GRACES', interpolation=True, interp_factor=3,
                 blaze_corr=False, blaze_corr_method='alphashape', useorders='all',
                 template_per_night=True):
        """

        Parameters
        ----------
        path : str
             Path where FITS files are stored.
        intrument : str, optional
             Which instrument the observations were taken with. Default is 'GRACES'.
        interpolation : bool, optional
             Whether or not to interpolate the spectrum onto a finer grid. Default is True.
        interp_factor : int, optional
             The factor by which to interpolate across. Default is 5
        blaze_corr : bool, optional
             Whether or not to correct for the blaze function. Default is False.
        blaze_corr_method : str, optional
             Which blaze correction method to use. Default is 'alphashape'.
        useorders : np.ndarray, optional
             Which orders to use in the analysis. Default is 'all'.
        template_per_night : bool, optional
             Allows users to create a template per each night of observations or
             across all observations. Default is True.
        
        Attributes
        ----------
        path : str
             Path where FITS files are stored.
        instrument : str
        fwhm : float
             FWHM value for the user's given instrument.
        """
        
        self.instrument = instrument
        self.lines_wave = None
        self.template_wave = None

        if path == None:
            print('Please input directory where files are located.')
            sys.exit()
        else:
            self.path = path
        
            # Sets instrument specific attributes
            ins = InstrumentDeets(instrument=instrument)
            attrs = [i for i in ins.__dict__.keys() if i[:1] != '_']
            for a in attrs:
                setattr(self, a, getattr(ins, a))

            self.load_files()
            self.remove_bad_pixels()

            if interpolation:
                lengths = []
                for uo in np.unique(self.orders[0]):
                    lengths.append(len(np.where(self.orders[0]==uo)[0]))
                factor = np.nanmax(lengths) * interp_factor

                self.wave, self.flux, self.flux_err, self.orders = interp_data(self.wave,
                                                                               self.flux,
                                                                               self.flux_err,
                                                                               self.orders,
                                                                               factor)
                self.wave *= self.wave_unit

            if blaze_corr:
                self.flux, self.flux_err = blaze_corr_func(self.wave,
                                                           self.flux,
                                                           self.flux_err,
                                                           self.orders,
                                                           blaze_corr_method)

            if useorders != 'all':
                self.remove_bad_orders(np.array(useorders))


    def load_files(self):
        """
        Loads in time and flux from given path files.
        The keys for searching the FITS files are set by the 
            instrument name.

        Attributes
        ----------
        wave : np.ndarray
        flux : np.ndarray
        flux_err : np.ndarray
        orders : np.ndarray
        times : astropy.time.core.Time
        airmass : np.ndarray
        """
        files = np.sort([os.path.join(self.path, i) for i in os.listdir(self.path) if i.endswith('.fits')])
        times = np.zeros(len(files))
        airmass = np.zeros(len(files))

        if len(files) == 0:
            print('There are no FITS files in this directory.')
            sys.exit()
        
        else:
            wave = np.zeros(len(files), dtype=np.ndarray)
            flux = np.zeros(len(files),dtype=np.ndarray)
            orders = np.zeros(len(files),dtype=np.ndarray)
            flux_err = np.zeros(len(files),dtype=np.ndarray)
        
            for i, fn in enumerate(files):

                hdu = fits.open(fn)
                
                times[i] = hdu[self.hdu_ext].header[self.time_key]
                airmass[i] = hdu[self.hdu_ext].header[self.airmass_key]

                w, f = np.array([]), np.array([])
                e, o = np.array([]), np.array([])

                for uo in np.unique(hdu[self.hdu_ext].data[self.order_key]):
                    q = hdu[self.hdu_ext].data[self.order_key] == uo
                    w = np.append(w, hdu[self.hdu_ext].data[self.wave_key][q][self.cut_order_ends:-self.cut_order_ends])
                    f = np.append(f, hdu[self.hdu_ext].data[self.flux_key][q][self.cut_order_ends:-self.cut_order_ends])
                    e = np.append(e, hdu[self.hdu_ext].data[self.flux_err_key][q][self.cut_order_ends:-self.cut_order_ends])
                    o = np.append(o, hdu[self.hdu_ext].data[self.order_key][q][self.cut_order_ends:-self.cut_order_ends])

                wave[i] = w
                flux[i] = f
                flux_err[i] = e
                orders[i] = o

                hdu.close()
                
            self.wave = wave
            self.flux = flux
            self.flux_err = flux_err
            self.orders = orders
            self.times = Time(times, format='mjd')
            self.airmass = airmass


    def remove_bad_orders(self, orders):
        """
        If useorders is an array, this function removes the unwanted orders.

        Parameters
        ----------
        orders : np.ndarray
        """
        neww = np.zeros(len(self.wave), dtype=np.ndarray)
        newf = np.zeros(len(self.wave),dtype=np.ndarray)
        newe = np.zeros(len(self.wave),dtype=np.ndarray)
        newo = np.zeros(len(self.wave),dtype=np.ndarray)
        
        for i in range(len(self.wave)):
            good_inds = np.array([], dtype=int)

            for o in np.unique(orders):
                good_inds = np.append(good_inds, np.where(self.orders[i]==o)[0])
        
            neww[i] = self.wave[i][good_inds]
            newf[i] = self.flux[i][good_inds]
            newe[i] = self.flux_err[i][good_inds]
            newo[i] = self.orders[i][good_inds]

        self.wave = neww
        self.flux = newf
        self.flux_err = newe
        self.orders = newo
    

    def remove_bad_pixels(self):
        """
        Masks regions with bad pixel flux values and
        regions with very deep lines.

        Returns
        -------
        time : np.ndarray
             Time array with masked bad values.
        flux : np.ndarray
             Flux array with masked bad values.
        flux_err : np.ndarray
             Flux error array with masked bad values.
        
        """
        ### HARDCODED NUMBER MAY NEED TO BE CHANGED
        ### 1.5 corresponds to value in Marshall's code for cosmic rays
        for i in range(len(self.flux)):
            f = self.flux[i] + 0.0
            bad = np.where( (self.flux[i] < 0) | 
                            (self.flux[i] > 1.5) )[0]

            self.wave[i] = np.delete(self.wave[i], bad)
            self.flux[i] = np.delete(self.flux[i], bad)
            self.flux_err[i] = np.delete(self.flux_err[i], bad)
            self.orders[i] = np.delete(self.orders[i], bad)


    def find_out_of_transit(self):
        """
        Identifies which indices are outside of the transit.

        Returns
        -------
        days : np.ndarray
             Rounded times to MJD days.
        when_dict : dictionary
             Indices of out-of-transit data for each unique 
             night to make template from.
        """
        days = np.array(self.times.value, dtype=int)

        when_dict = {}

        for j, uo in enumerate(np.unique(days)):
            out_inds = np.array([])

            for i in range(len(self.ephemeris)):
                arr = np.zeros(2)

                multiperiods = np.arange(1,1000,1) * self.period[i].value + self.ephemeris[i].mjd
                
                start = multiperiods - self.duration[i].to(u.day).value/2
                end   = multiperiods + self.duration[i].to(u.day).value/2

                # Sees if the start of an event is within the night
                try:
                    arr[0] = start[np.where(np.array(start,dtype=int)==uo)[0]]
                except:
                    arr[0] = -100
                    print("Start of object {1} transit begins before the night of {0} MJD.".format(uo, i))

                # Sees if the end of an event is within the night
                try:
                    arr[1] = end[np.where(np.array(end,dtype=int)==uo)[0]]
                except:
                    arr[1] = -100
                    print("Start of object {1} transit ends after the night of {0} MJD.".format(uo, i))

                # Finds indices which are out of the night for the event
                if arr[0] != -100 and arr[1] != -100:
                    good_inds = np.where((self.times.value < arr[0]) | 
                                         (self.times.value > arr[1]) )[0]
                elif arr[0] == -100 and arr[1] != -100:
                    good_inds = np.where(self.times.value > arr[1])[0]
                elif arr[0] != -100 and arr[1] == -100:
                    good_inds = np.where(self.times.value < arr[0])[0]

                out_inds = np.append(out_inds, good_inds)

            un, C = np.unique(out_inds, return_counts=True)
            out_inds = un[C>1]
            when_dict[uo] = out_inds

        return when_dict


    def create_template(self, ephemeris, duration, period,
                        ephemeris_units='MJD', 
                        duration_units=u.hr, template_per_night=True):
        """
        Creates a template per each order from each night.

        Parameters
        ----------
        ephemeris : float, np.ndarray
             Ephemeris/ephemerides of the object(s) transiting. This will be 
             used to determine which frames are out of transit to make the
             template.
        duration : float, np.ndarray
             Duration(s) of the object(s) transiting. This will be used to determine
             which frames are out of transit to make the template.
        period : float, np.ndarray
             Period(s) [in days] for the object(s) transiting. This will be used to determine
             which frames are out of the transit to make the template.
        ephemeris_units : str, optional
             The units the ephemeris/ephemerides is given in. Default is 'MJD'.
             Other options include astropy.time.Time formats. More information can be
             found here: https://docs.astropy.org/en/stable/time/#time-format.
        duration_units : astropy.units.Unit
             The units the duration is given in. Default is astropy.units.hour.
        template_per_night : bool, optional              
             Allows users to create a template per each night of observations or 
             across all observations. Default is True.    

        Attributes
        ----------
        ephemeris : astropy.time.Time
        duration : astropy.units.Quantity
        period : astropy.units.Quantity
        template_wave : np.ndarray
        template_flux : np.ndarray
        when_event_dict : dictionary
             The start and stop times for each event over a given night.
        """
        self.ephemeris = Time(ephemeris, format=ephemeris_units.lower())
        self.duration  = np.array(duration) * duration_units
        self.period = np.array(period) * u.day

        when_dict = self.find_out_of_transit()

        if template_per_night == True:
            per_day = len(when_dict)
        else:
            per_day = -1

        template_wave     = np.zeros((np.abs(per_day), len(self.wave[0])))
        template_flux     = np.zeros((np.abs(per_day), len(self.flux[0])))
        template_flux_err = np.zeros((np.abs(per_day), len(self.flux_err[0])))
        template_orders   = np.zeros((np.abs(per_day), len(self.orders[0])))

        if per_day == -1:
            inds = np.array([], dtype=int)
            for key in list(when_dict.keys()):
                inds = np.append(inds, when_dict[key])
            template_wave[i] = np.nanmean(self.wave[inds], axis=0)
            template_flux[i] = np.nanmedian(self.flux[inds], axis=0)
            template_flux_err[i] = np.sqrt( np.nansum( self.flux_err[inds]**2))/len(inds)
            template_orders[i] = np.nanmedian(self.orders[inds])

        else:
            for i in range(per_day):
                inds = np.array(when_dict[list(when_dict.keys())[i]], dtype=int)
                template_wave[i] = np.nanmean(self.wave[inds], axis=0)
                template_flux[i] = np.nanmedian(self.flux[inds], axis=0)
                template_flux_err[i] = np.sqrt( np.nansum( self.flux_err[inds]**2))/len(inds)
                template_orders[i] = np.nanmedian(self.orders[inds])


        ## THIS NEEDS TO BE TESTED FOR MULTI-DAY DATA SETS :D
        self.template_wave = template_wave
        self.template_flux = template_flux
        self.template_flux_err = template_flux_err
        self.template_orders = template_orders
        self.when_event_dict = when_dict
            

    def load_lines(self, filename, path=None, indices=[0,1],
                   units=u.Angstrom, delimiter=','):
        """
        Reads in file of lines to use. File should be a list
        of line wavelengths and depths at the bare minimum.

        Parameters
        ----------
        filename : str
             File name.
        path : str, optional
             Where the line file is stored. Default is
             current working directory.
        indices : np.ndarray, optional
             The indices for wavelength and depth in the lines file.
             Default is [0, 1].
        units : astropy.units.Unit, optional
             Unit of the line wavelength. Default is Angstroms.
        delimiter : str, optional
             Character(s) separating each column in the file. Default
             is ','.

        Attributes
        ----------
        lines_wave : np.ndarray
        lines_depth : np.ndarray
        """

        if path != None:
            filename = os.path.join(path, filename)
        
        data = np.loadtxt(filename, delimiter=delimiter)
        
        self.lines_wave = (data[:,indices[0]]*units).to(self.wave_unit)
        self.lines_depth = data[:,indices[1]]

        # Gets lines in our wavelength region
        ind = np.where( (self.lines_wave >= np.nanmin(self.wave[0])) &
                        (self.lines_wave <= np.nanmax(self.wave[0])) )[0]

        self.lines_wave = self.lines_wave[ind]
        self.lines_depth = self.lines_depth[ind]

    def extract_line_profiles(self, vsini, vsini_units=u.km/u.s):
        """
        Used to fit the lines! (Feels self-explanatory.) Basically
        calls a bunch of other functions.

        Parameters
        ----------
        vsini : float
             v*sin(i) of the star.
        vsini_units : astropy.units, optional
             Units of v*sin(i). Default is km/s.
        """
        if self.template_wave is None:
            print("A template needs to be created through DT.create_template()")
            return
        
        elif self.lines_wave is None:
            print("A file of specific lines needs to be read in through DT.load_lines()")
            return

        else:
            self.velocity_shift()
            self.depth_adjustment()
            self.line_shape()

    
    def velocity_shift(self):
        """
        The velocity offset between the model and spectrum
        is a free parameter. This is then subtracted off to put
        the spectra in the stellar rest frame.
        """
        return


    def depth_adjustment(self):
        """
        The depths of each spectral line is the free parameter,
        to compute an optimal model spectrum.
        """
        return
    

    def line_shape(self):
        """
        The velocity offset & line depth are fixed. The value of
        each pixel in the model line profile is the free parameter
        (i.e. fitting the shape of each line).
        """
        return



    def profile_constants(self, v, epsilon, vsini):
        """
        Constants needed for the profiles.
        """
        c1 = 2.0 * (1-epsilon)/(np.pi * vsini * (1-epsilon/3.0))
        c2 = 1.0 * epsilon/vsini * 1.0/(1-epsilon/3.0)

        dvdv = v / vsini

        Gv0 = c1 * np.sqrt(1 - dvdv**2) + c2 * (1-dvdv**2)
        Gv0 /= np.nanmax(Gv0)

        return c1, c2, dvdv, Gv0

    
    def profile_mp(self, vsini, epsilon, shift):
        """
        Parameters
        ----------
        vsini : float
        epsilon : float
        shift : float

        Returns
        -------
        rmsp : np.ndarray
             RMS value array (?) 
        """
        c1, c2, dvdv, Gv0 = self.profile_constants(epsilon, vsini)

        if (vsini < 0) or (epsilon > 1) or (epsilon < 0):
            rmsp = 10e5
        
        sigma = self.fwhm
        
        vi = v[np.abs(v) < 25.0]

        # Checks to make sure there are values in this array
        if len(vi) <= 1:
            vi = v[0:20] - v[10]

        instp = np.exp(-0.5 * (vi / sigma)**2)
        
        Gv = np.convolve(Gv0, instp)
        Gv /= np.nanmax(Gv)

        resids = self.avgprof - Gv
        rmsp = np.sqrt( np.nanmean(resids**2) )

        return rmsp
        


    def profile_1mp(self, v, vsini, epsilon=0.6):
        """
        Parameters
        ----------
        v : float
             Velocity.
        vsini : float
             v*sin(i) to inject into the profile.
        epislon : float, optional

        Returns
        -------
        Gv : np.ndarray
             Cconvolved velocity and instp.
        """
        
        c1, c2, dvdv, Gv0 = self.profile_constants(v, epsilon, vsini)

        if len( np.where( np.abs(dvdv) <= 1)[0]) <= 1:
            diff = dvdv[1] - dvdv[0]
            dvdv = len(v) - len(v)/2.0 + 0.5
            dvdv *= diff
            
        vi = vi[np.abs(v) <= 25]
        if len(vi) <= 1:
            vi = 21.0 - 10.0

        instp = np.exp(-0.5 * (vi / sigma)**2)

        Gv = np.convolve(Gv0, instp)
        Gv /= np.nanmax(Gv)
        
        return Gv
        
        

    def profile_2mp(self, vsini, epsilon, shift, v):
        """
        Parameters
        ----------
        vsini_n : float
        epsilon_in : float
        shift : float
        v : float
             Velocity.
             
        Returns
        -------
        Gv : np.ndarray
             Convolved velocity and instp.
        """

        c1, c2, dvdv, Gv0 = self.profile_constants(v+shift, epsilon, vsini)
        
        npix = len(v)

        if len(np.where(np.abs(dvdv) <= 1)[0]) <= 1 and self.instrument=='theor':
            print('WARNING: Marshall says, "oh no not again."')
            diff = dvdv[1] - dvdv[0]
            dvdv = npix - npix/2.0 + 0.5
            dvdv *= diff

        vi = vi[np.abs(v) <= 25]
        if len(vi) <= 1:
            vi = 21.0 - 10.0

        instp = np.exp(-0.5 * (vi / sigma)**2)

        Gv = np.convolve(Gv0, instp)
        Gv /= np.nanmax(Gv)

        return Gv
