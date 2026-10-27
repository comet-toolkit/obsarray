"""conventions - handling of the UNC metadata conventions version obsarray follows"""

import inspect
import os
import warnings

from obsarray.options import OPTIONS

__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"
__all__ = ["UNCConventionsWarning"]


UNC_CONVENTIONS_DOCS_URL = (
    "https://obsarray.readthedocs.io/en/latest/content/user/unc_conventions.html"
)

_UNC_CONVENTIONS_MESSAGE = (
    "obsarray currently writes uncertainty metadata following v0.1 of the UNC "
    "specification. A future release will default to UNC draft v0.2, which is being "
    "adopted into the CF conventions and is not backwards compatible with v0.1 (for "
    "example, 'unc_comps' becomes 'uncertainty_variables' and 'err_corr_*' becomes "
    "'error_correlation'). See " + UNC_CONVENTIONS_DOCS_URL + " for details. To keep "
    "the current behaviour and silence this warning, call "
    'obsarray.set_options(unc_conventions="v0.1").'
)

# set once the warning has been issued, so it is only shown once per session
_warned = False


class UNCConventionsWarning(FutureWarning):
    """
    Warning that obsarray's default UNC conventions version will change from v0.1 to draft v0.2

    Issued once per session, the first time obsarray writes uncertainty metadata, unless a conventions
    version has been chosen explicitly with :py:class:`~obsarray.options.set_options`. It can also be
    filtered like any other warning:

    .. code-block:: python

        import warnings
        import obsarray

        warnings.filterwarnings("ignore", category=obsarray.UNCConventionsWarning)
    """


def warn_unc_conventions() -> None:
    """
    Warns, once per session, that the default UNC conventions version will change - unless a version has
    been chosen explicitly with :py:class:`~obsarray.options.set_options`
    """

    global _warned

    if _warned or OPTIONS["unc_conventions"] is not None:
        return

    _warned = True
    warnings.warn(
        _UNC_CONVENTIONS_MESSAGE, UNCConventionsWarning, stacklevel=_find_stack_level()
    )


def _find_stack_level() -> int:
    """
    Returns the ``stacklevel`` of the first frame outside obsarray, so warnings point at the calling code

    :return: stack level to pass to :py:func:`warnings.warn` from the function calling this one
    """

    pkg_dir = os.path.dirname(os.path.abspath(__file__)) + os.sep

    frame = inspect.currentframe()
    level = 0
    while frame is not None and frame.f_code.co_filename.startswith(pkg_dir):
        frame = frame.f_back
        level += 1

    return level


if __name__ == "__main__":
    pass
