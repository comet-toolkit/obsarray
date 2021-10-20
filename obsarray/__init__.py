"""obsarray - Extension to xarray for handling uncertainty-quantified observation data"""

__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"
__all__ = []

from ._version import get_versions
from obsarray import accessor

__version__ = get_versions()["version"]
del get_versions

