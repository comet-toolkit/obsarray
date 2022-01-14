"""err_corr_forms - module for the defintion of error-correlation parameterisation forms"""

import abc
from typing import Callable, Type

__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"
__all__ = []


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

    def __init__(self):
        pass

    @abc.abstractmethod
    def form(self):
        pass


if __name__ == "__main__":
    pass
