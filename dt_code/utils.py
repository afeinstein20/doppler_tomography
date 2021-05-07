import numpy as np

__all__ = ['create_phases', 'hex_to_rgb']


def create_phases(time, vals):
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
    params.u = [vals[7], vals[8]]     #limb darkening coefficients [u1, u2]
    
    m = batman.TransitModel(params, time)
    lc = m.light_curve(params)
    
    tphase = np.zeros(len(lc))
    tphase[lc==1.0] = np.nan
    
    args = np.where(np.isnan(tphase)==False)[0]
    tphase[args] = np.linspace(0.0, 1.0, len(args))
    return tphase, lc


def hex_to_rgb(h):
    """ Converts hex to RGB colors. """
    if '#' in h:
        h = h.lstrip('#')   
    hlen = int(len(h))
    rgb = tuple(int(h[i:int(i+hlen/3)], 16) / 255.0 for i in range(0, hlen, int(hlen/3)))
    return rgb
