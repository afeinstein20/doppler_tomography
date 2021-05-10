import numpy as np
from astropy import units
from matplotlib import gridspec
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from .utils import create_phases, hex_to_rgb, discrete_cmap

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
        self.wavelengths = wavelengths
        self.spectra = spectra
        self.errors = errors
        self.orders = orders
        self.obstimes = obstimes
        
        self.phase = None

    def get_order_index(self, lam):
        """
        Loops over the wavelengths array and finds which index
        a given wavelength is located in.

        Parameters
        ----------
        lam : float
           The wavelength in question.
        
        Returns
        -------
        i : int
           The order index.
        """ 
        where = []
        centers = []
        for i in range(len(self.wavelengths[0])):
            diff = np.abs(self.wavelengths[0][i] - lam)
            if len(np.where(diff <= 1.0)[0]) > 0:
                where.append(i)
                c = np.where(self.wavelengths[0][i] <= lam)[0][-1]
                centers.append(np.abs(len(self.wavelengths[0][i])/2 - c))

        if len(where) > 1:
            return where[np.argmin(centers)]
        elif len(where) == 1:
            return where[0]
        else:
            return('Wavelength not in these observations.')


    def plotting_lines(self, reg, colors=None, ax=None, cmap='viridis',
                       xaxis='velocity'):
        """
        Plots lines over a given region.
        
        Parameters
        ----------
        xaxis : str
           Plots xaxis in wavelength or velocity
        """
        ind1 = self.get_order_index(reg[0])
        ind2 = self.get_order_index(reg[1])

        ind, q = self.line_mask(reg)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8,6))
        
        if colors is None:
            colors = discrete_cmap(cmap, len(self.spectra))

        if xaxis == 'velocity':
            vel,_ = self.to_velocity(self.wavelengths[0][ind][q],
                                     flux=self.spectra[0][ind][q])

            for i in range(len(self.spectra)):
                ax.plot(vel,
                        self.spectra[i][ind][q], c=colors[i])            
                
            ax.set_xlabel('Velocity [km s$^{-1}$]')

        else:
            for i in range(len(self.spectra)):
                ax.plot(self.wavelengths[i][ind][q],
                        self.spectra[i][ind][q], c=colors[i])
            ax.set_xlabel('Wavelength [nm]')

        ax.set_ylabel('Normalized Flux')

    def expanding_bins(self, reg, repeat=3, subtract=True, temptype='oot',
                       plot=False, ax=None, cmap='viridis', vlim=0.018,
                       vsini=0):
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
        ind1 = self.get_order_index(reg[0])
        ind2 = self.get_order_index(reg[1])

        ind, q = self.line_mask(reg)

        # Creates template
        template = self.create_template(temptype, ind)
        template = template[q] + 0.0

        binned = np.zeros( (len(self.obstimes)*repeat, len(self.spectra[0][ind][q]))  )
        rphase = np.zeros(len(self.obstimes)*repeat)
        z = 0
        for i in range(len(self.spectra)):
            if subtract == True:
                binned[z:z+repeat] = self.spectra[i][ind][q] - template
            else:
                binned[z:z+repeat] = self.spectra[i][ind][q]
            rphase[z:z+repeat] = self.phase[i]
            z += repeat

        # Plotting code
        if plot == True:
            if ax is None:
                fig, ax = plt.subplots(figsize=(8,5))

            im = ax.imshow(binned, cmap=cmap, vmin=-vlim,
                           vmax=vlim, origin='lower')
            cbar = plt.colorbar(im)
            cbar.set_label('Fractional Deviation')
            
            vel,_ = self.to_velocity(self.wavelengths[0][ind][q], 
                                     flux=self.spectra[0][ind][q])
            interp = interp1d(vel, np.arange(0,len(vel),1))
            
            ax.vlines(interp(-vsini), -10, len(binned)+30, color='k', lw=2.5, 
                       linestyle='--')
            ax.vlines(interp(vsini), -10, len(binned)+30, color='k', lw=2.5,
                             linestyle='--')
            ax.set_ylim(0, len(binned))

            xticks = [interp(-60), interp(0), interp(60)]
            ax.set_xticks(xticks)
            ax.set_xticklabels([-60, 0, 60])

            pinterp = interp1d(rphase, np.arange(0,len(rphase),1))
            for p in [0, 0.5, 1]:
                ax.plot(interp(0), pinterp(p), 'w+', ms=8, markeredgewidth=2,
                        markeredgecolor='w')
            yticks = np.arange(0,1.2,0.2)
            ax.set_yticks(pinterp(yticks))
            ax.set_yticklabels(np.round(yticks,1))
            
            ax.set_ylabel('Transit Phase')
            ax.set_xlabel('Velocity [km s$^{-1}$]')

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


    def transit_phases(self, params, ret=False):
        """
        Creates an array of transit phases based on the time of 
        observations and the batman transit modeling code.
        
        Parameters
        ----------
        params : np.ndarray
           T0, period, rp/rstar, a/rstar, i, e, omega, u1, u2.
           T0 must be given in JD.
        ret : bool, optional
           Returns the phase and light curve instead of setting
           attributes. Default is False.
        
        Attributes
        ----------
        phase : np.ndarray
           Transit phases.
        lc : np.ndarray
           Transit light curve.
        """
        phase, lc = create_phases(self.obstimes.jd, params)

        if ret == True:
            return phase, lc
        else:
            self.phase = phase
            self.lc = lc
        

    def create_template(self, temptype, ind):
        """ 
        Creates a template to subtract from the spectra. 
        temptype options = 'med' or 'oot' (out-of-transit).
        """
        if temptype == 'med':
            template = np.nanmedian(self.spectra[:,ind], axis=0)
        elif temptype == 'oot':
            if self.phase is None:
                return('Must call DTAnalysis.transit_phases() first.')
            else:
                oot = np.where(np.isnan(self.phase)==True)[0]
                template = np.nanmedian(self.spectra[oot,ind], axis=0)
        else:
            return('temptype keyword unknown.')
        return template


    def line_mask(self, reg):
        """
        Gets information on what order the wavelength is in.
        Returns : ind, mask
        """
        ind1 = self.get_order_index(reg[0])
        ind2 = self.get_order_index(reg[1])

        if ind1 == ind2:
            ind = ind1
        else:
            return('Line region spans two orders.')

        q = ( (self.wavelengths[0][ind] >= reg[0]) &
              (self.wavelengths[0][ind] <= reg[-1]) )

        return ind, q

    def measure_excess(self, reg, temptype='med', mask=False, sig=2.5):
        """
        Calculates excess absorption in a given line.
        
        Parameters
        ----------
        reg : np.ndarray
           2D array of lower and upper wavelengths to
           calculate excess over.
        temptype : str, optional
           The template to subtract from. Default is a median
           across all observations. The other option is 'oot'
           which is an out-of-transit median template.
        mask : bool, optional
           Can mask outlier points. Default is False.
        sig : float, optional
           Sigma outliers to mask if mask is True. Default is 2.5.

        Returns
        -------
        excess : np.ndarray
           Array of excess values.
        """
        ind1 = self.get_order_index(reg[0])
        ind2 = self.get_order_index(reg[1])

        ind, q = self.line_mask(reg)

        # Creates template
        template = self.create_template(temptype, ind)
        template = template[q] + 0.0 
        
        # Applies mask if mask==True
        if mask == True:
            widths = np.zeros(len(self.obstimes))
            divout = self.spectra[:,ind,q] / template
            errors = np.zeros(len(self.obstimes))

            for i in range(len(self.obstimes)):
                mask = np.where( divout[i] < np.nanmedian(divout[i]) + sig*np.nanstd(divout[i]) )[0]
                widths[i] = np.nansum( (self.spectra[i,ind,q][mask]/template[mask]) - 1 ) * -1.0
                errors[i] = 1.0 / np.nansum(self.errors[i,ind,q][mask])

        else:
            widths = np.nansum( (self.spectra[:,ind,q]/template - 1), axis=1 ) * -1.0
            errors = 1.0 / np.nansum( self.errors[:,ind,q]**2, axis=1)

        return widths, errors
        
        
    def weighted_means(self, reg):
        """
        Calculates weighted means for given lines.

        Parameters
        ----------
        reg : np.ndarray
           2 x n array of beginning and ending wavelength
           regions to sum up.
        
        Returns
        -------
        lc : np.ndarray
           Weighted mean light curve.
        err : np.ndarray
           Errors on the light curve.
        """
        n, d, e = 0.0, 0.0, 0.0

        reg = np.array(reg)

        if len(reg.shape) == 1:
            reg = np.reshape(reg, (1,2))

        for i in range(reg.shape[0]):
            ind, q = self.line_mask(reg[i])
            
            n += np.nansum( (self.spectra[:,ind,q] / self.errors[:,ind,q]**2) , axis=1)
            d += np.nansum( 1.0 / self.errors[:,ind,q]**2, axis=1)
            e += np.nansum( self.errors[:,ind,q]**-2, axis=1)

        e = np.sqrt(1.0/e)
        f = n/d

        return f, e
