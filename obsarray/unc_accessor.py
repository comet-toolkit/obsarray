"""unc_accessor - xarray extensions with accessor objects for uncertainty handling"""

from copy import deepcopy
import numpy as np
import xarray as xr
from typing import Union, Tuple, List, Optional
from comet_maths import convert_corr_to_cov, correlation_from_covariance
from obsarray.templater.template_util import create_var
from obsarray.templater.dataset_util import DatasetUtil
from obsarray.err_corr import err_corr_forms, BaseErrCorrForm
from obsarray.utils import empty_err_corr_matrix


__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"
__all__ = []


class Uncertainty:
    """
    Interface for handling ``xarray.Dataset`` uncertainty variables

    :param xarray_obj: dataset
    :param unc_var_name: name of uncertainty variable
    :param sli: slice of variable
    """

    def __init__(
        self,
        xarray_obj: xr.Dataset,
        var_name: str,
        unc_var_name: str,
        sli: Optional[tuple] = None,
    ):

        # initialise attributes

        self._obj = xarray_obj
        self._unc_var_name = unc_var_name
        self._var_name = var_name
        self._sli = tuple([slice(None)] * self._obj[self._unc_var_name].ndim)
        if sli is not None:
            self._sli = self._expand_sli(sli)

    def __str__(self):
        """Custom __str__"""
        return "<{}> \n{}".format(
            self.__class__.__name__,
            self.value.__repr__()[18:],
        )

    def __repr__(self):
        """Custom  __repr__"""
        return str(self)

    def __getitem__(self, sli: tuple):
        """
        Defines variable slice

        :param sli: slice of variable
        :return: self
        """

        # update slice
        self._sli = self._expand_sli(sli)

        return self

    def _expand_sli(self, sli: Optional[tuple] = None) -> tuple:
        """
        Function to expand the provided sli so that it always has the right number of dimensions

        :param sli: input sli tuple. This one can have fewer dimensions than the total if e.g. only providing the first index.
        :type sli: tuple
        :return: output sli tuple.
        :rtype: tuple
        """

        # if no slice provided, define as slice for full array
        if sli is None:
            out_sli = tuple([slice(None)] * self._obj[self._unc_var_name].ndim)

        # Otherwise, set each : dimension to slice(None)
        # E.g. if providing [0] for a variable with 3 dimensions, this becomes
        # [0,slice(None),slice(None)]
        else:
            out_sli = list([slice(None)] * self._obj[self._unc_var_name].ndim)
            sli_list = list(sli) if isinstance(sli, tuple) else [sli]
            for i in range(len(sli_list)):
                if not sli_list[i] == ":":
                    out_sli[i] = sli_list[i]
            out_sli = tuple(out_sli)

        return out_sli

    @property
    def err_corr(self) -> List[Tuple[Union[str, List[str]], BaseErrCorrForm]]:
        """
        Error-correlation parameterisation for uncertainty effect.

        Given as a list of parameterisations along different variable slice dimensions. Each parameterisation is given as a two-element tuple - where the first element is the dimension (or list of dimensions) along which the parameterisation applies, and the second element is the error-correlation parameterisation defining object (as subclass of ``obsarray.err_corr.BaseErrorCorrForm``).

        :return: Error correlation parameterisation
        """

        # Find dimensions in variable slice
        sli_dims = [
            dim
            for dim, idx in zip(self._obj.dims.keys(), self._sli)
            if not isinstance(idx, int)
        ]

        # Find number of error-correlation parameterisations along different dimensions in variable metadata
        err_corr_idxs = set(
            [
                a[9]
                for a in self._obj[self._unc_var_name].attrs.keys()
                if a[:8] == "err_corr"
            ]
        )

        # Loop through error-correlation parameterisations, extract metadata and build parameterisation
        # as a two element tuple (as described in doc string)
        err_corr = []
        for i in err_corr_idxs:

            dim_i = self._obj[self._unc_var_name].attrs[
                DatasetUtil.return_err_corr_dim_str(i)
            ]

            list_dim_i = deepcopy(dim_i)
            if isinstance(dim_i, str):
                list_dim_i = [dim_i]

            # continue if parmeterisation dimension in slice dimensions
            if [dim_i_j for dim_i_j in list_dim_i if dim_i_j in sli_dims]:

                form_i = self._obj[self._unc_var_name].attrs[
                    DatasetUtil.return_err_corr_form_str(i)
                ]

                params_i = self._obj[self._unc_var_name].attrs[
                    DatasetUtil.return_err_corr_params_str(i)
                ]

                units_i = self._obj[self._unc_var_name].attrs[
                    DatasetUtil.return_err_corr_units_str(i)
                ]

                err_corr.append(
                    (
                        dim_i,
                        err_corr_forms[form_i](
                            self._obj, self._unc_var_name, dim_i, params_i, units_i
                        ),
                    )
                )

        return err_corr

    @property
    def units(self) -> Union[str, None]:
        """
        Return uncertainty variable units

        :return: uncertainty variable units
        """
        return (
            self._obj[self._unc_var_name].attrs["units"]
            if "units" in self._obj[self._unc_var_name].attrs
            else None
        )

    @property
    def var_units(self) -> Union[str, None]:
        """
        Returns units of observation variable associated with uncertainty variable

        :return: observation variable units
        """
        return (
            self._obj[self._var_name].attrs["units"]
            if "units" in self._obj[self._var_name].attrs
            else None
        )

    @property
    def var_value(self) -> xr.DataArray:
        """
        Returns value of observation variable associated with uncertainty variable

        :return: observation variable
        """
        return self._obj[self._var_name][self._sli]

    @property
    def value(self) -> xr.DataArray:
        """
        Return uncertainty data array

        :return: uncertainty variable
        """
        return self._obj[self._unc_var_name][self._sli]

    @property
    def abs_value(self) -> xr.DataArray:
        """
        Returns uncertainty values with units that match those of the observation variable

        NB: only currently converts % to absolute, can't handle mismatched units

        :return: uncertainty values with same units as observation variable
        """

        if self.var_units is not None:
            if self.units == "%":
                return self.value / 100 * self.var_value

            elif (self.units != self.var_units) and (self.units is not None):
                raise ValueError(
                    "Unit mismatch between observation variable and uncertainty variable:\n"
                    "* '{}' - '{}'\n"
                    "* '{}' - '{}'".format(
                        self._var_name, self.var_units, self._unc_var_name, self.units
                    )
                )

        return self.value

    @property
    def pdf_shape(self):
        """
        Returns probability density function shape for uncertainty variable data

        :return: uncertainty variable pdf shape
        """
        return self._obj[self._unc_var_name].attrs["pdf_shape"]

    @property
    def is_random(self) -> bool:
        """
        Returns True if uncertainty is fully random in all dimensions

        :return: random uncertainty flag
        """
        if len(self.err_corr) > 0:
            return all(e[1].is_random is True for e in self.err_corr)
        else:
            return False

    @property
    def is_structured(self) -> bool:
        """
        Returns True if uncertainty is neither fully random or systematic in all dimensions

        :return: structured uncertainty flag
        """

        if all(e[1].is_random is True for e in self.err_corr) or all(
            e[1].is_systematic is True for e in self.err_corr
        ):
            return False

        return True

    @property
    def is_systematic(self) -> bool:
        """
        Returns True if uncertainty is fully systematic in all dimensions

        :return: systematic uncertainty flag
        """

        return all(e[1].is_systematic is True for e in self.err_corr)

    def err_corr_dict(self) -> dict:
        """
        Error-correlation dictionary for uncertainty effect.

        :return: dictionary with error-correlation for each dimension
        """

        # initialise error-correlation dictionary
        err_corr_dict = {}

        # populate with error-correlation matrices built be each error-correlation
        # parameterisation object
        for dim_err_corr in self.err_corr:
            if np.all(
                [
                    dim in self._obj[self._unc_var_name][self._sli].dims
                    for dim in dim_err_corr[1].dims
                ]
            ):
                if dim_err_corr[1].form in ["random", "systematic"]:
                    err_corr_dict[dim_err_corr[0]] = dim_err_corr[1].form

                elif dim_err_corr[1].form == "err_corr_matrix":
                    err_corr_dict[dim_err_corr[0]] = self._obj[
                        dim_err_corr[1].params[0]
                    ].values

                else:
                    raise NotImplementedError(
                        "this correlation form is not implemented for err_corr_dict()"
                    )
        return err_corr_dict

    def err_corr_dict_numdim(self) -> dict:
        """
        Error-correlation dictionary for uncertainty effect, where the keys are the dimension index rather than dimension name.

        :return: dictionary with error-correlation for each dimension
        """
        # initialise error-correlation dictionary
        err_corr_dict = self.err_corr_dict()
        err_corr_dict_numdim = {}

        for idim, dim in enumerate(self._obj.dims):
            if dim in err_corr_dict.keys():
                err_corr_dict_numdim[str(idim)] = err_corr_dict[dim]

        return err_corr_dict_numdim

    def err_corr_matrix(self) -> xr.DataArray:
        """
        Error-correlation matrix for uncertainty effect.

        :return: Error-correlation matrix
        """

        # initialise error-correlation matrix
        err_corr_matrix = empty_err_corr_matrix(
            self._obj[self._unc_var_name][self._sli]
        )

        # populate with error-correlation matrices built be each error-correlation
        # parameterisation object
        for dim_err_corr in self.err_corr:
            if np.all(
                [
                    dim in self._obj[self._unc_var_name][self._sli].dims
                    for dim in dim_err_corr[1].dims
                ]
            ):
                err_corr_matrix.values = err_corr_matrix.values.dot(
                    dim_err_corr[1].build_matrix(self._sli)
                )

        return err_corr_matrix

    def err_cov_matrix(self):
        """
        Error-covariance matrix for uncertainty effect

        :return: Error-covariance matrix
        """

        err_cov_matrix = empty_err_corr_matrix(self._obj[self._unc_var_name][self._sli])

        err_cov_matrix.values = convert_corr_to_cov(
            self.err_corr_matrix().values, self.value.values
        )

        return err_cov_matrix


class VariableUncertainty:
    """
    Interface for ``xarray.Dataset`` variable uncertainty handling

    :param xarray_obj: dataset
    :param var_name: name of dataset variable
    """

    def __init__(self, xarray_obj: xr.Dataset, var_name: str):
        self._obj = xarray_obj
        self._var_name = var_name
        self._sli = tuple([slice(None)] * self._obj[self._var_name].ndim)

    def __getitem__(
        self, key: Union[str, tuple]
    ) -> Union[Uncertainty, "VariableUncertainty"]:
        """
        Returns variable uncertainty interface

        :param key: uncertainty variable name or variable slice
        :return: uncertainty interface
        """

        if isinstance(key, str):
            return Uncertainty(self._obj, self._var_name, key, self._sli)

        self._sli = key
        return self

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

        self._obj.unc._add_unc_var(self._var_name, unc_var, unc_def)

    def __delitem__(self, unc_var):
        """
        Safely removes uncertainty variable

        :param unc_var: uncertainty variable name
        """
        self._obj.unc._remove_unc_var(self._var_name, unc_var)

    def __str__(self):
        """Custom __str__"""
        return "<{}>\nVariable Uncertainties: '{}'\n{}".format(
            self.__class__.__name__,
            self._var_name,
            self._obj.unc._var_unc_vars(self._var_name).__repr__(),
        )

    def __repr__(self):
        """Custom  __repr__"""
        return str(self)

    def __len__(self) -> int:
        """
        Returns number of variable uncertainties

        :returns: number of variable uncertainties
        """
        return len(self._obj.unc._var_unc_var_names(self._var_name))

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

    def keys(self) -> List[str]:
        """
        Returns uncertainty variable names

        :return: uncertainty variable names
        """
        return self._obj.unc._var_unc_var_names(self._var_name)

    @property
    def comps(self) -> xr.core.dataset.DataVariables:
        """
        Returns observation variable uncertainty data variables

        :return: uncertainty data variables
        """

        return self._obj.unc._var_unc_vars(self._var_name, self._sli)

    @property
    def random_comps(self) -> xr.core.dataset.DataVariables:
        """
        Returns dataset uncertainty data variables with fully random error-correlation

        :return: uncertainty data variables
        """

        return self._obj.unc._var_unc_vars(self._var_name, self._sli, unc_type="random")

    @property
    def structured_comps(self) -> xr.core.dataset.DataVariables:
        """
        Returns dataset uncertainty data variables with structured error-correlation

        :return: uncertainty data variables
        """

        return self._obj.unc._var_unc_vars(
            self._var_name, self._sli, unc_type="structured"
        )

    @property
    def systematic_comps(self) -> xr.core.dataset.DataVariables:
        """
        Returns dataset uncertainty data variables with fully systematic error-correlation

        :return: uncertainty data variables
        """

        return self._obj.unc._var_unc_vars(
            self._var_name, self._sli, unc_type="systematic"
        )

    def _quadsum_unc(self, unc_var_names):
        quadsum_unc = None
        for i, unc_var_name in enumerate(unc_var_names):
            if i == 0:
                quadsum_unc = deepcopy(self[unc_var_name].abs_value ** 2.0)
                quadsum_unc.name = None
            else:
                quadsum_unc += self[unc_var_name].abs_value ** 2.0

        if quadsum_unc is not None:
            quadsum_unc = quadsum_unc ** 0.5

        return quadsum_unc

    def total_unc(self) -> xr.DataArray:
        """
        Returns observation variable combined uncertainty for all uncertainty components (in absolute units)

        :return: total observation variable uncertainty
        """

        return self._quadsum_unc(self._obj.unc._var_unc_var_names(self._var_name))

    def random_unc(self) -> xr.DataArray:
        """
        Returns observation variable combined uncertainty for uncertainty components with fully random error-correlation

        :return: total random observation variable uncertainty
        """

        return self._quadsum_unc(
            self._obj.unc._var_unc_var_names(self._var_name, unc_type="random")
        )

    def structured_unc(self) -> xr.DataArray:
        """
        Returns observation variable combined uncertainty for uncertainty components with structured error-correlation

        :return: total structured observation variable uncertainty
        """

        return self._quadsum_unc(
            self._obj.unc._var_unc_var_names(self._var_name, unc_type="structured")
        )

    def systematic_unc(self) -> xr.DataArray:
        """
        Returns observation variable combined uncertainty for uncertainty components with fully systematic error-correlation

        :return: total systematic observation variable uncertainty
        """

        return self._quadsum_unc(
            self._obj.unc._var_unc_var_names(self._var_name, unc_type="systematic")
        )

    def total_err_corr_matrix(self) -> xr.DataArray:
        """
        Returns observation variable combined error-correlation matrix for all uncertainty components

        :return: total error-correlation matrix
        """

        total_err_corr_matrix = empty_err_corr_matrix(
            self._obj[self._var_name][self._sli]
        )

        total_err_corr_matrix.values = correlation_from_covariance(
            self.total_err_cov_matrix().values
        )

        return total_err_corr_matrix

    def structured_err_corr_matrix(self) -> xr.DataArray:
        """
        Returns observation variable combined error-correlation matrix for uncertainty components that do not have either fully random or fully systematic error-correlation

        :return: structured error-correlation matrix
        """

        structured_err_corr_matrix = empty_err_corr_matrix(
            self._obj[self._var_name][self._sli]
        )

        structured_err_corr_matrix.values = correlation_from_covariance(
            self.structured_err_cov_matrix().values
        )

        return structured_err_corr_matrix

    def total_err_cov_matrix(self) -> xr.DataArray:
        """
        Returns observation variable combined error-covariance matrix for all uncertainty components

        :return: total error-covariance matrix
        """

        total_err_cov_matrix = empty_err_corr_matrix(
            self._obj[self._var_name][self._sli]
        )
        covs_sum = np.zeros(total_err_cov_matrix.shape)
        for unc in self:
            covs_sum += unc[self._sli].err_cov_matrix().values

        total_err_cov_matrix.values = covs_sum

        return total_err_cov_matrix

    def structured_err_cov_matrix(self):
        """
        Returns observation variable combined error-covariance matrix for uncertainty components with structured error-correlation

        :return: structured error-covariance matrix
        """

        structured_err_cov_matrix = empty_err_corr_matrix(
            self._obj[self._var_name][self._sli]
        )

        covs_sum = np.zeros(structured_err_cov_matrix.shape)
        for unc in self:
            if unc.is_structured:
                covs_sum += unc[self._sli].err_cov_matrix().values

        structured_err_cov_matrix.values = covs_sum

        return structured_err_cov_matrix


@xr.register_dataset_accessor("unc")
class UncAccessor(object):
    """
    ``xarray.Dataset`` accesssor object for handling dataset variable uncertainties

    :param xarray_obj: xarray dataset
    """

    def __init__(self, xarray_obj: xr.Dataset):

        # Initialise attributes
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
        """
        Returns :py:class:`~obsarray.unc_accessor.VariableUncertainty` object to interface with uncertainty information for a given observation variable

        :param var_name: observation variable name
        :return: observation variable uncertainty interface
        """
        return VariableUncertainty(self._obj, var_name)

    def __len__(self):
        """
        Returns number of observation variables

        :return: number of observation variables
        """

        return len(self.keys())

    def __iter__(self) -> "UncAccessor":
        """
        Initialises iterator

        :return: self
        """
        self.i = 0  # Define counter
        return self

    def __next__(self):
        """
        Returns ith observation variable

        :return: ith obs variable
        """

        # Iterate through obs variables
        if self.i < len(self.keys()):
            self.i += 1  # Update counter
            return self[self.keys()[self.i - 1]]

        else:
            raise StopIteration

    def keys(self) -> list:
        """
        Returns observation variable names

        :return: observation variable names
        """

        return list(self._obj.unc.obs_vars.keys())

    @property
    def obs_vars(self) -> xr.core.dataset.DataVariables:
        """
        Returns dataset observation data variables (defined as dataset variables with uncertainties)

        :return: observation data variables
        """
        obs_var_names = []
        for var_name in self._obj.variables:
            if self._is_obs_var(var_name):
                obs_var_names.append(var_name)
        return self._obj[obs_var_names].data_vars

    @property
    def unc_vars(self) -> xr.core.dataset.DataVariables:
        """
        Returns dataset uncertainty data variables (defined as uncertainties associated with observation variables)

        :return: uncertainty data variables
        """
        unc_var_names = set()
        for var_name in self.obs_vars:
            var_unc_var_names = self._var_unc_var_names(var_name)

            if type(var_unc_var_names) == str:
                var_unc_var_names = [var_unc_var_names]

            unc_var_names |= set(var_unc_var_names)
        return self._obj[list(unc_var_names)].data_vars

    def _var_unc_var_names(
        self, obs_var_name: str, unc_type: Optional[str] = None
    ) -> List[str]:
        """
        Returns the names of uncertainty variables associated with specified observation variable

        :param obs_var_name: observation variable name
        :param unc_type: option to filter for specific uncertainty types, must have value ``"random"``, ``"structured"`` or ``"systematic"``

        :return: uncertainty variable names
        """

        # Get the names of unc vars defined in obs var metadata
        all_unc_var_names = []
        if "unc_comps" in self._obj[obs_var_name].attrs:
            all_unc_var_names = self._obj[obs_var_name].attrs["unc_comps"]
            if type(all_unc_var_names) == str:
                all_unc_var_names = [all_unc_var_names]

        # Filter returned names if unc_type defined
        unc_var_names = []
        if unc_type is None:
            unc_var_names = all_unc_var_names

        elif (
            (unc_type == "random")
            or (unc_type == "structured")
            or (unc_type == "systematic")
        ):
            for unc_var in all_unc_var_names:
                if getattr(self[obs_var_name][unc_var], "is_" + unc_type):
                    unc_var_names.append(unc_var)

        else:
            raise ValueError("no unc_type " + unc_type)

        return unc_var_names

    def _is_obs_var(self, var_name: str) -> bool:
        """
        Returns true if named dataset variable is an observation variable

        :return: observation variable flag
        """

        if self._var_unc_var_names(var_name):
            return True
        return False

    def _is_unc_var(self, var_name: str) -> bool:
        """
        Returns true if named dataset variable is an uncertainty variable

        :return: uncertainty variable flag
        """

        if var_name in self.unc_vars:
            return True
        return False

    def _var_unc_vars(
        self,
        obs_var_name: str,
        sli: Optional[tuple] = None,
        unc_type: Optional[str] = None,
    ) -> xr.core.dataset.DataVariables:
        """
        Returns uncertainty data variables for specified observation variable

        :return: uncertainty data variables
        """

        unc_var_names = self._var_unc_var_names(obs_var_name, unc_type=unc_type)

        if len(unc_var_names) == 0:
            return None

        if sli is None:
            return self._obj[unc_var_names].data_vars

        isel_sli = {dim: s for dim, s in zip(self._obj[unc_var_names[0]].dims, sli)}
        return self._obj[unc_var_names].isel(isel_sli).data_vars

    def _add_unc_var(
        self,
        obs_var: str,
        unc_var: str,
        unc_def: Union[xr.DataArray, Tuple[List[str], np.ndarray, Optional[dict]]],
    ) -> None:
        """
        Adds an uncertainty variable to the dataset

        :param obs_var: associated observation variable name
        :param unc_var: uncertainty variable name
        :param unc_def: either xarray DataArray/Variable, or definition through tuple as ``(dims, data[, attrs])``. ``dims`` is a list of variable dimension names, ``data`` is a numpy array of uncertainty values and ``attrs`` is a dictionary of variable attributes. ``attrs`` should include an element ``pdf_shape`` which defines the uncertainty probability density function form, and ``err_corr`` which defines the error-correlation structure of the data. If omitted ``pdf_shape`` is assumed Gaussian and the error-correlation is assumed random.
        """

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

    def _remove_unc_var(self, obs_var: str, unc_var: str) -> None:
        """
        Removes uncertainty variable from dataset

        :param obs_var: observation variable name
        :param unc_var: uncertainty variable name
        """

        del self._obj[unc_var]
        self._obj[obs_var].attrs["unc_comps"].remove(unc_var)


if __name__ == "__main__":
    pass
