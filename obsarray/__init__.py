"""obsarray - Extension to xarray for handling uncertainty-quantified observation data"""

__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"
__all__ = []

from ._version import get_versions
from obsarray import unc_accessor, flag_accessor
from obsarray.err_corr import err_corr_forms
from obsarray.templater.template_util import create_ds
from obsarray.templater.dstemplater import DSTemplater
from obsarray.templater.dswriter import DSWriter

__version__ = get_versions()["version"]
del get_versions
