"""err_corr_forms - module for the defintion of error-correlation parameterisation forms"""

import abc
from typing import Callable, Type, Union
import numpy as np
from comet_maths.linear_algebra.matrix_conversion import expand_errcorr_dims

__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"
__all__ = ["err_corr_forms", "register_err_corr_form", "BaseErrCorrForm"]


class ErrCorrForms:
    """
    Container for error-correlation parameterisation form definition objects
    """

    def __init__(self):
        self._forms = {}

    def __setitem__(self, form_name, form_cls):
        if not issubclass(form_cls, BaseErrCorrForm):
            raise TypeError(
                "form must be subclass of " + str(BaseErrCorrForm.__class__)
            )

        self._forms[form_name] = form_cls

    def __getitem__(self, name):
        return self._forms[name]

    def __delitem__(self, name):
        del self._forms[name]

    def keys(self):
        return self._forms.keys()


err_corr_forms = ErrCorrForms()


# placeholder currently
class BaseErrCorrForm(abc.ABC):
    """
    Base class for error-correlation parameterisation form defintions
    """

    is_random = False
    is_systematic = False

    def __init__(self, xarray_obj, unc_var_name, dims, params, units):
        self._obj = xarray_obj
        self._unc_var_name = unc_var_name
        self.dims = dims if isinstance(dims, list) else [dims]
        self.params = params if isinstance(params, list) else [params]
        self.units = units

    def __str__(self):
        """Custom __str__"""

        return self.form + " " + str(self.params)

    def __repr__(self):
        """Custom  __repr__"""
        return str(self)

    @property
    @abc.abstractmethod
    def form(self) -> str:
        """Form name"""
        pass

    def expand_dim_matrix(self, submatrix, sli):
        return expand_errcorr_dims(
            in_corr=submatrix,
            in_dim=self.dims,
            out_dim=list(self._obj[self._unc_var_name][sli].dims),
            dim_sizes={
                dim: self._obj.dims[dim]
                for dim in self._obj[self._unc_var_name][sli].dims
            },
        )

    def slice_full_cov(self, full_matrix, sli):
        mask_array = np.ones(self._obj[self._unc_var_name].shape, dtype=bool)
        mask_array[sli] = False

        return np.delete(
            np.delete(full_matrix, mask_array.ravel(), 0), mask_array.ravel(), 1
        )

    @abc.abstractmethod
    def build_matrix(self, sli: Union[np.ndarray, tuple]) -> np.ndarray:
        """
        Returns uncertainty effect error-correlation matrix, populated with error-correlation values defined
        in this parameterisation

        :param sli: slice of observation variable to return error-correlation matrix for

        :return: populated error-correlation matrix
        """
        pass


def register_err_corr_form(form_name: str) -> Callable:
    """
    Decorator for registering error-correlation parmaterisation form definition to package definition container

    :param form_name: name of error-correlation parmaterisation form e.g. "random"
    :return: decorator function
    """

    def decorator(form_cls: Type) -> Type:
        err_corr_forms[form_name] = form_cls
        return form_cls

    return decorator


@register_err_corr_form("random")
class RandomCorrelation(BaseErrCorrForm):

    form = "random"
    is_random = True

    def build_matrix(self, sli):
        """
        Returns uncertainty effect error-correlation matrix, populated with error-correlation values defined
        in this parameterisation

        :param sli: slice of observation variable to return error-correlation matrix for

        :return: populated error-correlation matrix
        """

        # evaluate correlation over matrices in form defintion
        dim_lens = [len(self._obj[dim]) for dim in self.dims]
        n_elems = int(np.prod(dim_lens))

        dims_matrix = np.eye(n_elems)

        # expand to correlation matrix over all variable dims
        return self.expand_dim_matrix(dims_matrix, sli)

        # # subset to slice
        # return self.slice_full_cov(full_matrix, sli)


@register_err_corr_form("systematic")
class SystematicCorrelation(BaseErrCorrForm):

    form = "systematic"
    is_systematic = True

    def build_matrix(self, sli):
        """
        Returns uncertainty effect error-correlation matrix, populated with error-correlation values defined
        in this parameterisation

        :param sli: slice of observation variable to return error-correlation matrix for

        :return: populated error-correlation matrix
        """

        # evaluate correlation over matrices in form defintion
        dim_lens = [len(self._obj[dim]) for dim in self.dims]
        n_elems = int(np.prod(dim_lens))

        dims_matrix = np.ones((n_elems, n_elems))

        # expand to correlation matrix over all variable dims
        return self.expand_dim_matrix(dims_matrix, sli)

        # subset to slice
        # return self.slice_full_cov(full_matrix, sli)


@register_err_corr_form("err_corr_matrix")
class ErrCorrMatrixCorrelation(BaseErrCorrForm):

    form = "err_corr_matrix"

    def build_matrix(self, sli):
        """
        Returns uncertainty effect error-correlation matrix, populated with error-correlation values defined
        in this parameterisation

        :param sli: slice of observation variable to return error-correlation matrix for

        :return: populated error-correlation matrix
        """

        # expand to correlation matrix over all variable dims
        return self.expand_dim_matrix(self._obj[self.params[0]], sli)

        # # subset to slice
        # return self.slice_full_cov(full_matrix, sli)


@register_err_corr_form("ensemble")
class EnsembleCorrelation(BaseErrCorrForm):

    form = "ensemble"

    def build_matrix(self, sli):
        """
        Returns uncertainty effect error-correlation matrix, populated with error-correlation values defined
        in this parameterisation

        :param sli: slice of observation variable to return error-correlation matrix for

        :return: populated error-correlation matrix
        """

        raise NotImplementedError


if __name__ == "__main__":
    pass
