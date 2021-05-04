import os
import sys
import emcee
import pickle
import corner
import numpy as np
from astropy import units
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from .horus import *

__all__ = ['LineProfileMCMC']

class LineProfileMCMC(object):
    
    """
    Uses the line profile code horus.py to fit planetary
    parameters. This class runs an MCMC to find the best
    fit parameters for your line shape.
    """

    def __init__(self, planet_name, times, exptime, wavelengths, spectra, error, 
                 velocities=None, wavelength_unit=units.nm, transit_phase=None,
                 planet_parameters=None):
        """

        Parameters
        ----------
        planet_name : str
           Official name of the planet.
        times : np.ndarray
           Array of observation times.
        exptime : np.ndarray
           Array of exposure times for each obervation.
        wavelengths : np.ndarray
           Must be the same shape as spectra. Should be relatively centered
           on the given line in question.
        spectra : np.ndarray
           Must be the same shape as wavelengths. Should be relatively centered
           on the given line in question.
        error : np.ndarray
           Must be the same shape as wavelengths/spectra.
        velocities : np.ndarray, optional
           The corresponding velocity array for the wavelengths. Default is None.
           If None, a velocities array will be created and centered around the
           minimum flux in a median spectral template.
        wavelength_unit : astropy.units.Unit
           The unit the wavelength array is given in. Default is
           nanometers (astropy.units.nm).
        transit_phase : np.ndarray, optional
           Array of planet phases that corresponds to the times array. Default is
           None. If no phases are input, then the phase array will be created using
           an array of planet parameters and the batman transit modeling code.
        planet_paramters : np.ndarray, optional
           An array of planet parameters used to create an array of transit phases
           during the observations given. Required if the transit_phases is not passed in.
           Array should be structured as -- [T0, Period, Rp/Rstar, a/Rstar, inclination,
           eccentricity, omega, u1, u2] where u1 and u2 are the quadratic limb darkening
           parameters.
        """
        self.planet_name = planet_name
        self.times = times
        self.exptime = exptime
        self.wavelengths = wavelengths * wavelength_unit
        self.spectra = spectra
        self.error = error

        if self.velocities is None:
            self.velocities = self.to_velocity()
        else:
            self.velocities = velocities

        if transit_phase is None:
            self.transit_phase = self.create_phases(planet_parameters)

        self.pickle_input = self.build_pickle()
        
        
        def to_velocity(self):
            """
            Transforms wavelengths to velocity in km / s. Centers
            the line around the minimum in the spectrum.
            """
            argmid = np.argmin(self.oot_template)
            
            lambda0 = self.wavelengths[0][argmid] + 0.0
            rv_m_s = ( (self.wavelengths - lambda0) / lambda0 * 3e8)*units.m/units.s
            self.velocities = rv_m_s.to(units.km/units.s)


        def create_phases(self, vals):
            """
            Creates a transit light curve over the input times. Used to back-out
            the transit phases for the given observations.
            """
            import batman

            params = batman.TransitParams()   #object to store transit parameters
            params.t0 = vals[0]               #time of inferior conjunction
            params.per = vals[1]              #orbital period
            params.rp = vals[2]               #planet radius (in units of stellar radii)
            params.a = vals[3]                #semi-major axis (in units of stellar radii)
            params.inc = vals[4]              #orbital inclination (in degrees)
            params.ecc = vals[5]              #eccentricity
            params.w = vals[6]                #longitude of periastron (in degrees)
            params.limb_dark = "quadratic"    #limb darkening model
            params.u = [vals[7], vals[8]]     #limb darkening coefficients [u1, u2, u3, u4]
            
            m = batman.TransitModel(params, self.times)    #initializes model
            lc = m.light_curve(params)

            tphase = np.zeros(len(lc))
            tphase[lc==1.0] = np.nan
            
            args = np.where(np.isnan(tphase)==False)[0]
            tphase[args] = np.linspace(0.0, 1.0, len(args))
            self.transit_phase = tphase
            return
            
            
        def build_pickle(self, mask): 
            """
            Build the data dictionary needed for horus.py.
            
            Parameters
            ----------
            mask : np.ndarray
               Mask used on the velocity/spectra arrays.

            Returns
            -------
            dic : dictionary
            """
            if mask is None:
                mask = np.arange(0, len(self.velocities), 1, dtype=int)

            dic = {'ttime':self.times,
                   'texptime':self.exptime,
                   'vabsfine':vabsfine[mask],
                   'profarr':profarr[mask],
                   'avgprof':np.nanmedian(profarr, axis=0)[mask],
                   'profarrerr':profarrerr[mask],
                   'avgproferr':np.nanmedian(profarrerr, axis=0)[mask],
                   'whichplanet':whichplanet}

            return dic


        def build_struct(data, params):
            # data = dictionary
            # Params = array of [vsini, period, lambda, b, rplanet, e, omega, a, gamma1, gamma2, intwidth]
            tomphase = np.mod(data['ttime']-params[12], params[1])
            highphase = np.where(tomphase >= params[1]/2)[0]
            tomphase[highphase] -= params[1]
            tomphase = (tomphase*units.day).to(units.min)
            
            horus_struc = {'vsini': params[0], 
                           'sysname': 'v1298tau_c', 
                           'obs': 'geminin1', 
                           'vabsfine': data['vabsfine']+params[11], 
                           'profarr': data['profarr'],
                           'Pd': params[1], 
                           'lambda': params[2], 
                           'b': params[3],
                           'rplanet': params[4],
                           't': tomphase.value, 
                           'times': data['ttime'], 
                           'e': params[5], 
                           'periarg': params[6]*np.pi/180., 
                           'a': params[7],
                           'gamma1': params[8],
                           'gamma2': params[9],
                           'width': params[10]}
            return horus_struc

        def upside_down_gauss(self, x, mu, sig, f):
            """
            Creates an upside-down Gaussian to fit for line cores. 

            Parameters
            ----------
            x : np.ndarray
               Array of velocities to evaluate the Gaussian over.
            mu : float
               Mean of the Gaussian.
            sig : float
               Standard deviation of the Gaussian.
            f : float
               Factor by which to scale the Gaussian.

            Returns
            -------
            model : np.ndarray
               Gaussian model.
            """
            term1 = f / (sig * np.sqrt(2 * np.pi))
            e = -0.5 * (x - mu)**2 / sig**2
            return -term1 * np.exp(e)

        def build_horus_model(self, hdict, params, x, linetype, symmetric=True):
            """
            Calls horus.py and builds the line profile to fit the data to.

            Parameters
            ----------
            hdict : dictionary
               Dictionary format to feed into the horus.py script.
            params : np.ndarray
               Array of parameters to fit in the line profile.
            x : np.ndarray
               Array of velocities to evaluate the line profile over.
            linetype : str
               Type of line to fit. Options are 'core' and 'line'.
            symmetric : bool, optional
               If linetype == 'core', this key tells if the Gaussian should
               be symmetric about the core or offset. Default is True.               
            """
            new_out = HorusModel(hdict, resnum=50, convol='y', 
                                  add_poly=False)
                
            # mu, std, factor
            if linetype == 'core':
                if symmetric == True:
                    model = self.upside_down_gauss(x, -params[11],
                                                   params[14], 
                                                   params[15])+params[13]
                else:
                    model = self.upside_down_gauss(x, -params[17],
                                                   params[14], 
                                                   params[15])+params[13]
        
            else:
                model = np.zeros(len(new_out['profarr'][0]))+params[13]


            return model + new_out['profarr'][0]*params[14]


        def log_prior(self, params, bounds, linetype, symmetric):
            """
            Prior function
            """
                                     
            if params[5] < 0:
                return -np.inf
            else:
                e = np.sqrt(params[5]) * np.sin(params[6])
                o = np.sqrt(params[5]) * np.cos(params[6])
                
                if params[8] < bounds[8][0] or params[8] > bounds[8][1]: #u1
                    return -np.inf
                if params[9] < bounds[9][0] or params[9] > bounds[9][1]: # u2
                    return -np.inf    
                if params[2] < bounds[2][0] or params[2] > bounds[2][1]: # obliquity
                    return -np.inf
                if e < 0.0 or e > 0.5: #eccentricity
                    return -np.inf
                if o < -np.pi or o > np.pi: #omega
                    return -np.inf
                if params[13] < bounds[13][0] or params[13] > bounds[13][1]: #y-offset 
                    return -np.inf
                if params[14] <= bounds[14][0] or params[14] > bounds[14][1]: #core scaling
                    return -np.inf
                if params[11] < bounds[11][0] or params[11] > bounds[11][1]: #line center
                    return -np.inf
                if params[10] < bounds[10][0] or params[10] > bounds[10][1]: 
                    return -np.inf
            
            if linetype=='core':
                if params[15] < bounds[15][0] or params[15] > bounds[15][1]: #gauss std
                    return -np.inf
                if params[16] < bounds[16][0] or params[16] > bounds[16][1]: #gauss scale factor
                    return -np.inf
                
                if symmetric==False:
                    if params[17] < bounds[17][0] or params[17] > bounds[17][1]: #gauss mean
                        return -np.inf

            # visini, period, b, rp/rstar, a/rstar, T0
            g_inds = [0, 1, 3, 4, 7, 12]

            means = [ bounds[0][0], bounds[1][0], bounds[3][0], 
                      bounds[4][0], bounds[7][0], bounds[12][0] ]
#            stds = [1.0, 1e-5, 0.05, 1e-5, 0.01, 1e-5]
            stds = [ bounds[0][1], bounds[1][1], bounds[3][1],
                     bounds[4][1], bounds[7][1], bounds[12][1] ]
            tracker = 0
            
            for i, g in enumerate(g_inds):
                term1 = 1.0 / (stds[i] * np.sqrt(2 * np.pi))
                e = (params[g] - means[i])**2 / (2.0*stds[i]**2)
                tracker += np.log(term1 * np.exp(-e))
            return tracker


        def lnlike(self, params, bounds, data, linetype, symmetric):
            """ Log-likelihood function """
            lp = self.log_prior(params, bounds, linetype, symmetric)
            
            if not np.isfinite(lp):
                return -np.inf
            else:
                hdict = self.build_struct(data, params)
                model = self.build_horus_model(hdict, params, data['vabsfine'])
                
                lnl = np.nansum( (profile-model)**2 / error**2 )
                return -0.5 * lnl



        def run_mcmc(self, nwalkers, nsteps, init_guess, 
                     bounds, output_key, linetype='core', symmetric=True, 
                     mask=None):
            """

            Parameters
            ----------
            nwalkers : int
               Number of walkers for the MCMC.
            nsteps : int
               Number of steps for the MCMC.
            init_guess : np.ndarray
               Initial guess of model parameters.
            bounds : np.ndarray
               Bounds within to accept each parameter. Needs to be of
               shape (len(init_guess), 2) where each entry is the pair
               of lower and upper bounds for uniform priors and the 
               (mean, std) for Gaussian priors.
            output_key : str
               Key to add to saved output file.
            linestype : str, optional
               Tells the type of line to fit. Default is 'core'. 
               Other optionl is 'line'. The difference is when set
               to 'core' an upside down Gaussian is added to the
               profile to better fit the profile.
            symmetric : bool, optional
               Tells if the underlying Gaussian around a line core 
               should be symmetrical or not. Default is True.
            mask : np.ndarray, optional
               An array to mask the data on. Default is None.
            """
            new_dict = self.build_pickle(mask=mask)
            
            hdict = self.build_struct(new_dict, starters)
            model = self.build_horus_model(hdict, starters, new_dict['vabsfine'])

            np.random.seed(321)

            pos = np.zeros((nwalkers, len(init_guess)))
            for i in range(len(init_guess)):
                pos[:,i] = np.random.uniform(init_guess[i]-bounds[i]/10.0,
                                             init_guess[i]+bounds[i]/10.0, nwalkers)

            nwalkers, ndim = pos.shape

            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnlike, args=(bounds,
                                                                               new_dict,
                                                                               linetype,
                                                                               symmetric))
            mcmc_output = sampler.run_mcmc(pos, nsteps, progress=True)
            self.sampler = mcmc_output
            self.chains = sampler.get_chain()
            self.accp_frac = sampler.acceptance_fraction
            self.lnprob = sampler.lnprobability

            np.save('samples_{}.npy'.format(output_key), sampler.get_chain())
            np.save('accp_frac_{}.npy'.format(output_key), sampler.acceptance_fraction)
            np.save('lnprob_{}.npy'.format(output_key), sampler.lnprobability)
        
"""
if key == 'caII':
    starters = [23.0, 8.24958, 5.0, 0.200060, 0.039208, 0.1, 0.0, 13.19, 0.37, 0.3, 0.3, 
                4.0, 58846.097156, 0.97, 45.0, 33.0, 0.275]
    limits = [5.0, 1e-5, 3.0, 0.05, 1e-5, 0.01, 5.0, 0.01, 0.1, 0.1, 0.1, 
              0.1, 1e-5, 0.2, 10.0, 10.0, 0.2]
elif key == 'caI':
    starters = [23.0, 8.24958, 5.0, 0.200060, 0.039208, 0.1, 0.0, 13.19, 0.37, 0.3, 0.3, 
                8.5, 58846.097156, 1.04, 45.0, 30.0, 0.18, 13.0]
    limits = [5.0, 1e-5, 3.0, 0.05, 1e-5, 0.01, 5.0, 0.01, 0.1, 0.1, 0.1, 
              0.1, 1e-5, 0.2, 10.0, 10.0, 0.2, 0.1]
elif key == 'caIII':
    starters = [23.0, 8.24958, 5.0, 0.200060, 0.039208, 0.1, 0.0, 13.19, 0.37, 0.3, 0.3, 
                0.5, 58846.097156, 0.99, 48.0, 35.0, 0.15, 3.0]
    limits = [5.0, 1e-5, 3.0, 0.05, 1e-5, 0.01, 5.0, 0.01, 0.1, 0.1, 0.1, 
              0.1, 1e-5, 0.2, 10.0, 10.0, 0.2, 0.1]

"""
