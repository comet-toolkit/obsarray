"""unc_accessor - xarray extensions with accessor objects for uncertainty handling"""

import numpy as np
import xarray as xr
from typing import Union, Tuple, List, Optional
from obsarray.templater.template_util import create_var
from obsarray.templater.dataset_util import DatasetUtil


__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"
__all__ = []


class Err:
    def __init__(self, xarray_obj, unc_var_name):
        self._obj = xarray_obj
        self.unc_var_name = unc_var_name

    @property
    def params(self):

        err_corr_idxs = set(
            [
                a[9]
                for a in self._obj[self.unc_var_name].attrs.keys()
                if a[:8] == "err_corr"
            ]
        )

        err_corr = []
        for i in err_corr_idxs:
            err_corr.append(
                {
                    "dim": self._obj[self.unc_var_name].attrs[
                        DatasetUtil.return_err_corr_dim_str(i)
                    ],
                    "form": self._obj[self.unc_var_name].attrs[
                        DatasetUtil.return_err_corr_form_str(i)
                    ],
                    "params": self._obj[self.unc_var_name].attrs[
                        DatasetUtil.return_err_corr_params_str(i)
                    ],
                    "units": self._obj[self.unc_var_name].attrs[
                        DatasetUtil.return_err_corr_units_str(i)
                    ],
                }
            )

        return err_corr

    @property
    def param_dims(self) -> list:
        return [p["dim"] for p in self.params]

    def dim_params(self, dim: Union[str, List[str]]) -> list:

        # if dim in self.param_dims:
        #     raise ValueError("dim must be one of " + str(self.param_dims))

        params = self.params

        if dim is not None:
            params = [p for p in self.params if p["dim"] == dim][0]

        return params

    def _params_to_err_corr_matrix(self, dim=None):
        params = self.dim_params(dim)

        len_dims = self._eval_len_dims(dim)

        if params["form"] == "random":
            return np.eye(len_dims)

        elif params["form"] == "systematic":
            return np.ones((len_dims, len_dims))

        elif params["form"] == "custom":
            return self._obj[params["params"][0]]

    def _eval_len_dims(self, dim):
        if type(dim) == str:
            dim = [dim]

        len_dims = 0
        for d in dim:
            len_dims += len(self._obj[self.unc_var_name][d])

        return len_dims


class ErrCorr(Err):
    def __str__(self):
        """Custom __str__"""

        params_string = ""
        for p in self.params:
            params_string += "* " + p.__repr__()[1:-1] + "\n"

        return "<{} '{}'> \n{}".format(
            self.__class__.__name__, self.unc_var_name, params_string
        )

    def __repr__(self):
        """Custom  __repr__"""
        return str(self)

    def to_matrix(self, dim):
        return self._params_to_err_corr_matrix(dim)


class ErrCov(Err):
    pass

    # def to_matrix(self, dim):
    #     err_corr = self._params_to_err_corr_matrix(dim)


class Uncertainty:
    """
    Interface for handling ``xarray.Dataset`` uncertainty variables

    :param xarray_obj: dataset
    :param unc_var_name: name of uncertainty variable
    """

    def __init__(self, xarray_obj, unc_var_name):
        self._obj = xarray_obj
        self.unc_var_name = unc_var_name

    def __str__(self):
        """Custom __str__"""
        return "<{}> \n{}".format(
            self.__class__.__name__,
            self.data.__repr__()[18:],
        )

    def __repr__(self):
        """Custom  __repr__"""
        return str(self)

    @property
    def data(self) -> xr.DataArray:
        """
        Return uncertainty data array

        :return: uncertainty variable
        """
        return self._obj[self.unc_var_name]

    @property
    def pdf_shape(self):
        """
        Returns probability density function shape for uncertainty variable data

        :return: uncertainty variable pdf shape
        """
        return self._obj[self.unc_var_name].attrs["pdf_shape"]

    @property
    def err_corr(self) -> ErrCorr:
        """
        Returns error correlation interface object

        :return: error correlation interface
        """
        return ErrCorr(self._obj, self.unc_var_name)

    @property
    def err_cov(self) -> ErrCov:
        """
        Returns error covariance interface object

        :return: error covariance interface
        """
        return ErrCov(self._obj, self.unc_var_name)

    def _get_err_corr_forms(self):
        forms = set()
        for dim_params in self.err_corr.params:
            forms.add(dim_params["form"])

        return list(forms)

    @property
    def is_random(self):
        """
        Returns True if uncertainty is fully random in all dimensions

        :return: random uncertainty bool
        """

        if self._get_err_corr_forms() == ["random"]:
            return True
        return False

    @property
    def is_systematic(self):
        """
        Returns True if uncertainty is fully systematic in all dimensions

        :return: systematic uncertainty bool
        """

        if self._get_err_corr_forms() == ["systematic"]:
            return True
        return False


class VariableUncertainty:
    """
    Interface for ``xarray.Dataset`` variable uncertainty handling

    :param xarray_obj: dataset
    :param var_name: name of dataset variable
    """

    def __init__(self, xarray_obj: xr.Dataset, var_name: str):
        self._obj = xarray_obj
        self.var_name = var_name

    def __getitem__(self, unc_var: str) -> Uncertainty:
        """
        Returns variable uncertainty interface

        :param unc_var: uncertainty variable name
        :return: uncertainty interface
        """
        return Uncertainty(self._obj, unc_var)

    def __setitem__(
        self,
        unc_var: str,
        unc_def: Union[
            xr.DataArray, xr.Variable, Tuple[List[str], np.ndarray, Optional[dict]]
        ],
    ):
        """
        Adds defined uncertainty variable to dataset

        :param unc_var: uncertainty variable name
        :param unc_def: either xarray DataArray/Variable, or definition through tuple as ``(dims, data[, attrs])``. ``dims`` is a list of variable dimension names, ``data`` is a numpy array of uncertainty values and ``attrs`` is a dictionary of variable attributes. ``attrs`` should include an element ``pdf_shape`` which defines the uncertainty probability density function form, and ``err_corr`` which defines the error-correlation structure of the data. If omitted ``pdf_shape`` is assumed Gaussian and the error-correlation is assumed random.

        :return: uncertainty variable interface
        """

        self._obj.unc._add_unc_var(self.var_name, unc_var, unc_def)

    def __delitem__(self, unc_var):
        """
        Safely removes uncertainty variable

        :param unc_var: uncertainty variable name
        """
        self._obj.unc._remove_unc_var(self.var_name, unc_var)

    def __str__(self):
        """Custom __str__"""
        return "<{}>\nVariable Uncertainties: '{}'\n{}".format(
            self.__class__.__name__,
            self.var_name,
            self._obj.unc._var_unc_vars(self.var_name).__repr__(),
        )

    def __repr__(self):
        """Custom  __repr__"""
        return str(self)

    def __len__(self) -> int:
        """
        Custom  __len__

        :returns: number of variable uncertainties
        """
        return len(self._obj._var_unc_vars(self.var_name))

    def __iter__(self):
        """Custom  __iter__"""

        self.i = 0  # Define counter
        return self

    def __next__(self) -> Uncertainty:
        """
        Returns ith variable uncertainty

        :return: uncertainty variable
        """

        # Iterate through uncertainty comp
        if self.i < len(self.keys()):
            self.i += 1  # Update counter
            return self[self.keys()[self.i - 1]]

        else:
            raise StopIteration

    @property
    def comps(self):
        return self._obj.unc._var_unc_vars(self.var_name)

    @property
    def random_comps(self):
        random_comp_names = []

        for unc_var_name in self.keys():
            if self[unc_var_name].is_random:
                random_comp_names.append(unc_var_name)

        return self._obj[random_comp_names].data_vars

    @property
    def systematic_comps(self):
        systematic_comp_names = []

        for unc_var_name in self.keys():
            if self[unc_var_name].is_systematic:
                systematic_comp_names.append(unc_var_name)

        return self._obj[systematic_comp_names].data_vars

    def keys(self):
        return list(self.comps.keys())

    @property
    def total(self):
        return self.comps._dataset.unc._quadsum()

    @property
    def random(self):
        return self.random_comps._dataset.unc._quadsum()

    @property
    def systematic(self):
        return self.systematic_comps._dataset.unc._quadsum()


@xr.register_dataset_accessor("unc")
class UncAccessor(object):
    """
    ``xarray.Dataset`` accesssor object for handling dataset variable uncertainties

    :param xarray_obj: xarray dataset
    """

    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj

    def __str__(self) -> str:
        """Custom __str__"""

        string = "<{}>\nDataset Uncertainties:\n".format(self.__class__.__name__)

        for var_name in self.keys():
            string += "* {}\n{}\n".format(
                var_name, self._obj.unc._var_unc_vars(var_name).__repr__()[16:]
            )

        return string

    def __repr__(self) -> str:
        """Custom  __repr__"""
        return str(self)

    def __getitem__(self, var_name: str) -> VariableUncertainty:
        """Custom  __repr__"""
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
            var_unc_var_names = self._var_unc_var_names(var_name)

            if type(var_unc_var_names) == str:
                var_unc_var_names = [var_unc_var_names]

            unc_var_names |= set(var_unc_var_names)
        return self._obj[list(unc_var_names)].data_vars

    def _var_unc_var_names(self, obs_var_name):

        unc_var_names = []
        if "unc_comps" in self._obj[obs_var_name].attrs:
            unc_var_names = self._obj[obs_var_name].attrs["unc_comps"]
            if type(unc_var_names) == str:
                unc_var_names = [unc_var_names]

        return unc_var_names

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
            if type(self._obj[obs_var].attrs["unc_comps"]) == str:
                self._obj[obs_var].attrs["unc_comps"] = [
                    self._obj[obs_var].attrs["unc_comps"]
                ]

            self._obj[obs_var].attrs["unc_comps"].append(unc_var)

        else:
            self._obj[obs_var].attrs["unc_comps"] = unc_var

    def _remove_unc_var(self, obs_var, unc_var):
        del self._obj[unc_var]
        self._obj[obs_var].attrs["unc_comps"].remove(unc_var)

    def _quadsum(self):
        return sum(d for d in (self._obj ** 2).data_vars.values()) ** 0.5


if __name__ == "__main__":
    pass
