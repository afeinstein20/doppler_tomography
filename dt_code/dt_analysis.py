import os
import numpy as np
from astropy import units
from astropy.io import fits
from astropy.time import Time
from astropy.table import Table
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord

from .observatories import Observatories

__all__ = ['DopplerTomography']

class DopplerTomography(object):
    """
    A tested approach for Doppler Tomography analysis
    of young, active stars with GRACES data.
    """

    def __init__(self, fn_dir, dattype_keyword='OBSTYPE', 
                 time_keyword='MJDATE', time_format='mjd',
                 reload=False):
        """

        Parameters
        ----------
        fn_dir : path
           The directory where all raw data files are stored.
        dattype_keyword : str, optional                
           Header keyword for the observation type.
           Default is 'OBSTYPE'.  
        time_keyword : str, optional              
           Header keyword for the time of observation.
           Default is 'MJDATE'.   
        time_format : str, optional
           Header time format. Default is 'mjd'. Options
           can be found in astropy.time.Time objects.   
        reload : bool, optional

        Attributes
        ----------
        fn_dir : path
        science_frames : np.ndarray
        med_dark : np.ndarray
        med_flat : np.ndarray
        """

        self.fn_dir = fn_dir

        if reload == False:
            self.resave(dattype_keyword, time_keyword, time_format)
            self.med_dark = self.create_master_files(np.sort([i for i in self.npy_files if
                                                              i.endswith('_BIAS.npy')]))
            np.save(os.path.join(self.fn_dir, 'master_dark.npy'), self.med_dark)

            self.med_flat = self.create_master_files(np.sort([i for i in self.npy_files if
                                                              i.endswith('_FLAT.npy')]))
            np.save(os.path.join(self.fn_dir, 'master_flat.npy'), self.med_flat)

        else:
            self.npy_files = np.sort([os.path.join(self.fn_dir, i) for i in
                                      os.listdir(self.fn_dir)])
            self.reload_master_frames() 

            
        self.science_frames = self.load_science_frames()


    def resave(self, dattype_keyword, time_keyword, time_format):
        """
        Resaves FITS files into .npy files
        and extracts observation times.

        Parameters
        ----------
        dattype_keyword : str
           Header keyword for the observation type.
           Default is 'OBSTYPE'.
        time_keyword : str
           Header keyword for the time of observation.
           Default is 'MJDATE'.
        time_format : str
           Header time format. Default is 'mjd'. Options
           can be found in astropy.time.Time objects.


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
            
            for fn in files:
                name = fn.split('.')[0]
                
                hdu = fits.open(fn)
                dattype = hdu[0].header[dattype_keyword]

                newname = '{0}_{1}.npy'.format(name, dattype)
                npy_files = np.append(npy_files, newname)
                
                np.save(newname, hdu[0].data)

                if dattype == 'OBJECT':
                    times = np.append(times, hdu[0].header[time_keyword])

            self.times = Time(times, format=time_format).jd
            np.save(os.path.join(self.fn_dir, 'jd_times.npy'), self.times)

            self.npy_files = npy_files
            return
        else:
            return("No FITS files found in this directory.")
        

    def reload_master_frames(self):
        """
        Reloads pre-saved master dark and flat files.

        Attributes
        ----------
        times : np.ndarray
        med_dark : np.ndarray
        med_flat : np.ndarray
        """
        
        self.times = Time(np.load(os.path.join(self.fn_dir, 'jd_times.npy')),
                          format='jd')
        self.med_dark = np.load(os.path.join(self.fn_dir,
                                             'master_dark.npy'))
        self.med_flat = np.load(os.path.join(self.fn_dir,
                                             'master_flat.npy'))
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


    def barycentric_correction(self, target, observatory='keck'):
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
        obs = Observatories(observatory)
        
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
                                                            location=obs.geodetic).to(units.km/units.s).value
        self.barycorr = barycorr
