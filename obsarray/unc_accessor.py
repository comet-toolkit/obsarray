"""unc_accessor - xarray extensions with accessor objects for uncertainty handling"""

import numpy as np
import xarray as xr
from typing import Union, Tuple, List, Optional
from obsarray.templater.template_util import create_var

__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"
__all__ = []


@xr.register_dataset_accessor("unc")
class UncAccessor(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __getitem__(self, var_name):
        return VariableUncertainty(self._obj, var_name)

    def __len__(self):
        return len(self.keys())

    def __iter__(self):
        self.i = 0  # Define counter
        return self

    def __next__(self):
        """
        Returns ith function
        """

        # Iterate through obs variables
        if self.i < len(self.keys()):
            self.i += 1  # Update counter
            return self[self.keys()[self.i - 1]]

        else:
            raise StopIteration

    def keys(self):
        return list(self._obj.unc.obs_vars.keys())

    @property
    def obs_vars(self):
        obs_var_names = []
        for var_name in self._obj.variables:
            if self._is_obs_var(var_name):
                obs_var_names.append(var_name)
        return self._obj[obs_var_names].data_vars

    @property
    def unc_vars(self):
        unc_var_names = set()
        for var_name in self.obs_vars:
            unc_var_names |= set(self._var_unc_var_names(var_name))
        return self._obj[list(unc_var_names)].data_vars

    def _var_unc_var_names(self, obs_var_name):
        return (
            self._obj[obs_var_name].attrs["unc_comps"]
            if "unc_comps" in self._obj[obs_var_name].attrs
            else []
        )

    def _is_obs_var(self, var_name):
        if self._var_unc_var_names(var_name):
            return True
        return False

    def _is_unc_var(self, var_name):
        if var_name in self.unc_vars:
            return True
        return False

    def _var_unc_vars(self, obs_var_name):
        return self._obj[self._var_unc_var_names(obs_var_name)].data_vars

    def _add_unc_var(
        self,
        obs_var: str,
        unc_var: str,
        unc_def: Union[xr.DataArray, Tuple[List[str], np.ndarray, Optional[dict]]],
    ):

        # add uncertainty variable
        if type(unc_def) == xr.DataArray:
            self._obj[unc_var] = unc_def

        # use templater functionality if var defined with tuple
        elif type(unc_def) == tuple:

            attrs = {}
            if len(unc_def) == 3:
                attrs = unc_def[2]

            if "err_corr" not in attrs:
                attrs["err_corr"] = []

            var_attrs = {
                "dtype": unc_def[1].dtype,
                "dim": unc_def[0],
                "attributes": attrs,
            }

            size = {d: s for d, s in zip(unc_def[0], unc_def[1].shape)}

            var = create_var(unc_var, var_attrs, size)
            var.values = unc_def[1]

            self._obj[unc_var] = var

        # add variable to uncertainty components attribute
        if self._is_obs_var(obs_var):
            self._obj[obs_var].attrs["unc_comps"].append(unc_var)
        else:
            self._obj[obs_var].attrs["unc_comps"] = [unc_var]

    def _remove_unc_var(self, obs_var, unc_var):
        del self._obj[unc_var]
        self._obj[obs_var].attrs["unc_comps"].remove(unc_var)

    def _quadsum(self):
        return sum(d for d in (self._obj ** 2).data_vars.values()) ** 0.5


class VariableUncertainty:
    """
    Interface for observation variable uncertainty handling
    """

    def __init__(self, xarray_obj, var_name):

        if not xarray_obj.unc._is_obs_var(var_name):
            raise ValueError("no uncertainty variables for " + var_name)

        self._obj = xarray_obj
        self.var_name = var_name

    def __getitem__(self, unc_var: str):
        return self._obj._var_unc_vars(self.var_name)

    def __setitem__(
        self,
        unc_var: str,
        unc_def: Union[xr.DataArray, Tuple[List[str], np.ndarray, Optional[dict]]],
    ):
        self._obj.unc._add_unc_var(self.var_name, unc_var, unc_def)

    def __delitem__(self, unc_var):
        self._obj.unc._remove_unc_var(self.var_name, unc_var)

    def __str__(self):
        """Custom __str__"""
        return "<{}>:\nVariable Uncertainties: '{}'\n{}".format(
            self.__class__.__name__,
            self.var_name,
            self._obj.unc._var_unc_vars(self.var_name).__repr__(),
        )

    def __repr__(self):
        """Custom  __repr__"""
        return str(self)

    def __len__(self):
        return len(self._obj._var_unc_vars(self.var_name))

    def __iter__(self):
        self.i = 0  # Define counter
        return self

    def __next__(self):
        """
        Returns ith function
        """

        # Iterate through uncertainty comp
        if self.i < len(self.keys()):
            self.i += 1  # Update counter
            return self[self.keys()[self.i - 1]]

        else:
            raise StopIteration

    def keys(self):
        return list(self._obj.unc._var_unc_vars(self.var_name).keys())

    @property
    def total(self):
        return self._obj.unc._var_unc_vars(self.var_name)._dataset.unc.quadsum()

    @property
    def comps(self):
        return self._obj.unc._var_unc_vars(self.var_name)


@xr.register_dataarray_accessor("err_corr")
class ErrCorrAccessor(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def _to_dict(self):
        err_corr = {}

        err_corr_attrs = {
            attr: val for attr, val in self._obj.items if attr[:8] == "err_corr"
        }

        n_dims_err_corr = None


if __name__ == "__main__":
    pass
