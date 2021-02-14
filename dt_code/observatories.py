from astropy import units as u
from astropy.coordinates import EarthLocation

__all__ = ['Observatories']

class Observatories(object):

    def __init__(self, observatory):
        """
        Gets latitude, longitude, and height for 
        calculating barycentric correction for different
        observatories.
        """

        if observatory.lower() == 'keck':
            self.get_keck()

        else:
            return('Observatory information not available yet.')


    def get_keck(self):
        self.lat = 19.8283 * u.deg
        self.lon = -155.4783 * u.deg
        self.height = 4160 * u.m

        self.geodetic = EarthLocation.from_geodetic(lat=self.lat,
                                                    lon=self.lon,
                                                    height=self.height)
