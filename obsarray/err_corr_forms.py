"""err_corr_forms - module for the defintion of error-correlation parameterisation forms"""

import abc
from typing import Callable, Type
import numpy as np

__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"
__all__ = ["err_corr_forms"]


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

    def __init__(self, xarray_obj, unc_var_name, params, units):
        self._obj = xarray_obj
        self.unc_var_name = unc_var_name
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

    @abc.abstractmethod
    def build_matrix(self, idx: np.ndarray) -> np.ndarray:
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
class RandomForm(BaseErrCorrForm):

    form = "random"

    def build_matrix(self, x):
        return np.eye(len(x))


@register_err_corr_form("systematic")
class SystematicForm(BaseErrCorrForm):

    form = "systematic"

    def build_matrix(self, idx):
        return np.ones((len(idx), len(idx)))


@register_err_corr_form("custom")
class CustomForm(BaseErrCorrForm):

    form = "custom"

    def build_matrix(self, idx):
        return self._obj[self.params[0]]


if __name__ == "__main__":
    pass
