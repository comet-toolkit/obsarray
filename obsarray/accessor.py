"""accessor - xarray extensions with accessor objects"""

import xarray as xr

__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"
__all__ = []


@xr.register_dataset_accessor("unc")
class UncAccessor(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def var_unc_var_names(self, obs_var_name):
        return self._obj[obs_var_name].attrs["u_components"] if "u_components" in self._obj[obs_var_name].attrs else []

    def is_obs_var(self, var_name):
        if self.var_unc_var_names(var_name):
            return True
        return False

    @property
    def obs_vars(self):
        obs_var_names = []
        for var_name in self._obj.variables:
            if self.is_obs_var(var_name):
                obs_var_names.append(var_name)
        return self._obj[obs_var_names].data_vars

    @property
    def unc_vars(self):
        unc_var_names = set()
        for var_name in self.obs_vars:
            unc_var_names |= set(self.var_unc_var_names(var_name))
        return self._obj[list(unc_var_names)].data_vars

    def is_unc_var(self, var_name):
        if var_name in self.unc_vars:
            return True
        return False

    def var_unc_vars(self, obs_var_name):
        return self._obj[self.var_unc_var_names(obs_var_name)].data_vars

    def quadsum(self):
        return sum(d for d in (self._obj ** 2).data_vars.values()) ** 0.5

    def u_tot(self, obs_var):
        return self.var_unc_vars(obs_var)._dataset.unc.quadsum()

    def add_unc_var(self, obs_var, unc_def):
        self._obj[unc_def[0]] = unc_def[1]

        if self.is_obs_var(obs_var):
            self._obj[obs_var].attrs["u_components"].append(unc_def[0])
        else:
            self._obj[obs_var].attrs["u_components"] = [unc_def[0]]


if __name__ == "__main__":
    pass
