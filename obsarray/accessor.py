"""accessor - xarray extensions with accessor objects"""

import xarray as xr

__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"
__all__ = []


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


if __name__ == "__main__":
    pass
