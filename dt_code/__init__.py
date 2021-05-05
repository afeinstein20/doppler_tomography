import os
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

from .version import __version__
from .observatories import *
from .instruments import *
from .reduction import *
#from .horus_mcmc import *
#from .horus import *
