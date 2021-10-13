"""obsarray - Extension to xarray for handling uncertainty-quantified observation data"""

__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"
__all__ = []

from ._version import get_versions
import xarray as xr

__version__ = get_versions()["version"]
del get_versions


@xr.register_dataset_accessor("unc")
class UncAccessor(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._unc = {}

    def set(self, name, value):
        if name in self._obj.keys():
            self._unc[name] = value

    def get(self, name):
        if (name in self._obj.keys()) and (name in self._unc.keys()):
            return self._unc[name]

