"""accessor - xarray extensions with accessor objects"""

import xarray as xr

__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"
__all__ = []


@xr.register_dataset_accessor("unc")
class UncAccessor(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    @property
    def measured_variables(self):
        measured_variables = []
        for v in self._obj.variables:
            if "u_components" in self._obj[v].attrs:
                measured_variables.append(v)
        return measured_variables

    @property
    def uncertainty_variables(self):
        u_components = set()
        for v in self._obj.variables:
            if "u_components" in self._obj[v].attrs:
                u_components |= set(self._obj[v].attrs["u_components"])
        return list(u_components)


if __name__ == "__main__":
    pass
