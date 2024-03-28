"""err_corr_forms - module for the defintion of error-correlation parameterisation forms"""

import abc
from typing import Callable, Type, Union, List
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

    def get_varshape_errcorr(self):
        """
        return shape of uncertainty variable, including only dimensions which are included in the current error correlation form.

        :return: shape of included dimensions
        """
        all_dims = self._obj[self._unc_var_name].dims
        all_dims_sizes = self._obj.sizes

        return tuple([all_dims_sizes[dim] for dim in all_dims if dim in self.dims])

    def get_sliced_dim_sizes_uncvar(self, sli: tuple) -> dict:
        """
        Return dictionary with sizes of sliced dimensions of unc variable, including all dimensions.

        :param sli: slice (tuple with slice for each dimension)
        :return: dictionary with shape of included sliced dimensions
        """
        uncvar_dims = self._obj[self._unc_var_name][sli].dims
        uncvar_shape = self._obj[self._unc_var_name][sli].shape
        return {
            uncvar_dims[idim]: uncvar_shape[idim] for idim in range(len(uncvar_dims))
        }

    def get_sliced_dim_sizes_errcorr(self, sli: tuple) -> dict:
        """
        Return dictionary with sizes of sliced dimensions of unc variable, including only dimensions which are
        included in the current error correlation form.

        :param sli: slice (tuple with slice for each dimension)
        :return: dictionary with shape of included sliced dimensions
        """
        uncvar_sizes = self.get_sliced_dim_sizes_uncvar(sli)
        sliced_dims = self.get_sliced_dims_errcorr(sli)

        return {dim: uncvar_sizes[dim] for dim in sliced_dims}

    def get_sliced_dims_errcorr(self, sli: tuple) -> list:
        """
        Return dimensions which are within the slice and included in the current error correlation form.

        :param sli: slice (tuple with slice for each dimension)
        :return: list with sliced dimensions
        """
        all_dims = self._obj[self._unc_var_name].dims
        return [
            all_dims[idim]
            for idim in range(len(all_dims))
            if (isinstance(sli[idim], slice) and all_dims[idim] in self.dims)
        ]

    def get_sliced_shape_errcorr(self, sli: tuple) -> tuple:
        """
        return shape of sliced uncertainty variable, including only dimensions which are included in the current error correlation form.

        :param sli: slice (tuple with slice for each dimension)
        :return: shape of included sliced dimensions
        """
        uncvar_sizes = self.get_sliced_dim_sizes_uncvar(sli)
        sliced_dims = self.get_sliced_dims_errcorr(sli)

        return tuple([uncvar_sizes[dim] for dim in sliced_dims])

    def slice_errcorr_matrix(self, err_corr_matrix, variable_shape, sli) -> np.ndarray:
        """
        Slice the provided error correlation matrix (typically the error correlation matrix of the
        BaseErrCorrForm) using the

        :param err_corr_matrix: error correlation matrix to be sliced
        :param variable_shape: tuple with the length of the dimensions in the error correlation matrix (in correct order for flattening)
        :param sli: slice of observation variable to return error-correlation matrix for
        :return: sliced error correlation matrix
        """
        mask_array = np.ones(variable_shape, dtype=bool)
        mask_array[sli] = False

        return np.delete(
            np.delete(err_corr_matrix, mask_array.ravel(), 0), mask_array.ravel(), 1
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

    def build_dot_matrix(self, sli: Union[np.ndarray, tuple]) -> np.ndarray:
        """
        Returns expanded error correlation matrix for use in dot product with error correlation
        in other dimensions.

        The (sliced) error correlation matrix for this BaseErrCorrForm is expanded from its current
        (sliced) dimensions (which often don't include all dimensions of the associated uncertainty
        variable) to the dimensions of the full (sliced) error correlation (i.e. all dimensions of
        the uncertainty).
        The returned matrix is not meaningfull unless combined in a dot product with the expanded
        matrices of other error correlation matrices (together spanning all uncertainty dimensions).

        :param sli: slice of observation variable to return error-correlation matrix for
        :return: expanded matrix for use in dot product with error correlation in other dimensions.
        """
        return expand_errcorr_dims(
            in_corr=self.build_matrix(sli),
            in_dim=self.get_sliced_dims_errcorr(sli),
            out_dim=list(self._obj[self._unc_var_name][sli].dims),
            dim_sizes=self.get_sliced_dim_sizes_uncvar(sli),
        )


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

    def build_matrix(self, sli: tuple) -> np.ndarray:
        """
        Returns uncertainty effect error-correlation matrix, populated with error-correlation values defined
        in this parameterisation

        :param sli: slice of observation variable to return error-correlation matrix for

        :return: populated error-correlation matrix
        """

        # evaluate correlation over matrices in form defintion
        dim_lens = self.get_sliced_shape_errcorr(sli)
        n_elems = int(np.prod(dim_lens))

        submatrix = np.eye(n_elems)

        return submatrix


@register_err_corr_form("systematic")
class SystematicCorrelation(BaseErrCorrForm):

    form = "systematic"
    is_systematic = True

    def build_matrix(self, sli: tuple) -> np.ndarray:
        """
        Returns uncertainty effect error-correlation matrix, populated with error-correlation values defined
        in this parameterisation

        :param sli: slice of observation variable to return error-correlation matrix for

        :return: populated error-correlation matrix
        """

        # evaluate correlation over matrices in form defintion
        dim_lens = self.get_sliced_shape_errcorr(sli)
        n_elems = int(np.prod(dim_lens))

        submatrix = np.ones((n_elems, n_elems))

        return submatrix


@register_err_corr_form("err_corr_matrix")
class ErrCorrMatrixCorrelation(BaseErrCorrForm):

    form = "err_corr_matrix"

    def build_matrix(self, sli: tuple) -> np.ndarray:
        """
        Returns uncertainty effect error-correlation matrix, populated with error-correlation values defined
        in this parameterisation

        :param sli: slice of observation variable to return error-correlation matrix for

        :return: populated error-correlation matrix
        """
        all_dims = self._obj[self._unc_var_name].dims

        sli_submatrix = tuple(
            [sli[i] for i in range(len(all_dims)) if all_dims[i] in self.dims]
        )

        submatrix = self.slice_errcorr_matrix(
            self._obj[self.params[0]], self.get_varshape_errcorr(), sli_submatrix
        )

        return submatrix


@register_err_corr_form("ensemble")
class EnsembleCorrelation(BaseErrCorrForm):

    form = "ensemble"

    def build_matrix(self, sli: tuple) -> np.ndarray:
        """
        Returns uncertainty effect error-correlation matrix, populated with error-correlation values defined
        in this parameterisation

        :param sli: slice of observation variable to return error-correlation matrix for

        :return: populated error-correlation matrix
        """

        raise NotImplementedError


if __name__ == "__main__":
    pass
