import alphashape
import numpy as np
from scipy.interpolate import interp1d

def order_start_end(w, o):
    """
    Getting start and end wavelengths per each order.
    This is necessary for some, but not all, observations.

    Parameters
    ----------
    w : np.ndarray
         Wavelength array.
    o : np.ndarray
         Orders array.

    Returns
    -------
    start_end : np.ndarray
         2D array of start and end wavelengths per order.
    """
    start_end = np.zeros((len(np.unique(o[0])), 2))

    for i, uo in enumerate(np.unique(o[0])):
        temp_start = np.zeros(len(w))
        temp_stop  = np.zeros(len(w))
        
        for j in range(len(w)):
            q = o[j] == uo
            temp_start[j] = w[j][q][0]
            temp_stop[j]  = w[j][q][-1]

        start_end[i][0] = np.nanmax(temp_start)
        start_end[i][1] = np.nanmin(temp_stop)

    return start_end

def interp_data(w, f, e, o, factor_per_order):
    """
    Interpolates spectrum onto a finer wavelength grid.

    Parameters
    ----------
    w : np.ndarray
         Wavelength array.
    f : np.ndarray
         Spectrum array.
    e : np.ndarray
         Spectrum error array.
    o : np.ndarray
         Orders array.
    factor_per_order : int
         Interpolation factor per order.

    Returns
    -------
    wave : np.ndarray
         Finer wavelength grid array.
    flux : np.ndarray
         Finer spectrum grid array.
    flux_err : np.ndarray
         Finer spectrum error grid array.
    orders : np.ndarray
         Finer orders grid array.
    """

    start_end = order_start_end(w, o)

    interp_w = np.zeros( ( len(w), factor_per_order * len(np.unique(o[0])) ))
    interp_f = np.zeros( ( len(w), factor_per_order * len(np.unique(o[0])) ))
    interp_e = np.zeros( ( len(w), factor_per_order * len(np.unique(o[0])) ))
    interp_o = np.zeros( ( len(w), factor_per_order * len(np.unique(o[0])) ))

    for i in range(len(w)):
        start = 0
        
        for j, uo in enumerate(np.unique(o[i])):

            q = o[i] == uo
        
            redstart = start_end[j][0]
            blueend  = start_end[j][1]

            finer_wavelength = np.logspace(np.log10(redstart),
                                           np.log10(blueend),
                                           factor_per_order,
                                           base=10.0)
            finer_wavelength[0] = redstart
            finer_wavelength[-1] = blueend

            func = interp1d(w[i][q], f[i][q])
            efunc = interp1d(w[i][q], e[i][q])

            interp_o[i][start:int(start+factor_per_order)] = uo
            interp_w[i][start:int(start+factor_per_order)] = finer_wavelength
            interp_f[i][start:int(start+factor_per_order)] = func(finer_wavelength)
            interp_e[i][start:int(start+factor_per_order)] = efunc(finer_wavelength)

            start += factor_per_order
            
    return interp_w, interp_f, interp_e, interp_o
            

    

def fit_alphashape(w, f, deg=11, alpha=2.5, mask=None):
    """
    Applies the alphashape approach to fit the blaze function.

    Parameters
    ----------
    w : np.ndarray
         Wavelength array.
    f : np.ndarray
         Flux array.
    deg : int
         Polynomial degree to smooth alpha-shape.
    alpha : float
    
    mask : np.ndarray, optional
         Indices to mask.

    Returns
    -------
    alpha_shape : np.ndarray
    y_smooth : np.ndarray
         The smoothed function to remove the blaze.
    bound_wave : 
    bound_spec : 
    """
    if mask is not None:
        w_new = w[mask] + 0.0
        f_new = f[mask] + 0.0
    else:
        w_new = w + 0.0
        f_new = f + 0.0
        
    points = [(w_new[i], f_new[i]) for i in range(len(w_new))]
    
    alpha_shape = alphashape.alphashape(points, alpha)

    try:
        indices = [np.argmin(alpha_shape.exterior.xy[0]), 
                           np.argmax(alpha_shape.exterior.xy[0])]
        if indices[0] > indices[1]:
            indices = [indices[0], len(alpha_shape.exterior.xy[0])]
        bound_wave = alpha_shape.exterior.xy[0][indices[0]:indices[1]]
        bound_spec = alpha_shape.exterior.xy[1][indices[0]:indices[1]]
    except:
        indices = [np.argmin(alpha_shape.boundary.geoms[0].xy[0]),
                   np.argmax(alpha_shape.boundary.geoms[0].xy[0])]
        if indices[0] > indices[1]:
            indices = [indices[0], len(alpha_shape.boundary.geoms[0].xy[0])]

        bound_wave = alpha_shape.boundary.geoms[0].xy[0][indices[0]:indices[1]]
        bound_spec = alpha_shape.boundary.geoms[0].xy[1][indices[0]:indices[1]]

    fit = Polynomial.fit(bound_wave, bound_spec, deg=deg)
    y_smooth = fit(w)

    return alpha_shape, y_smooth, bound_wave, bound_spec


def blaze_corr_func(wave, flux, flux_err, orders, method='alphashape'):
    """
    Applies a blaze correction on an order-by-order basis.

    Parameters
    ----------
    wave : np.ndarray
    flux : np.ndarray
    flux_err : np.ndarray
    orders : np.ndarray
    method : str, optional
         Which blaze correction method to use. Default is 'alphashape'.
    """
    
    for i in range(len(wave)):
        
        for uo in np.unique(orders[i]):
            
            q = np.where(orders[i] == uo)[0]

            # normalizing term
            u = ( np.nanmax(wave[i][q]) - np.nanmin(wave[i][q]) ) / (10 * np.nanmax(flux[i][q]))
            f = flux[i][q] * u

            # Fit 1 alpha-shape
            std1 = 3.0
            mask = np.where( (f < std1*np.std(f) + np.nanmedian(f)) &
                             (f > -std1*np.nanstd(f) + np.nanmedian(f)))[0]

            alpha_shape, y_smooth, bw, bs = fit_alpha(w, f, alpha=2.5, mask=mask)

            flux[i][q] = f / y_smooth

    return flux, flux_err
