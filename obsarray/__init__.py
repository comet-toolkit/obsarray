"""obsarray - Extension to xarray for handling uncertainty-quantified observation data"""

__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"
__all__ = []

from importlib.metadata import PackageNotFoundError, version

from obsarray import unc_accessor, flag_accessor
from obsarray.err_corr import err_corr_forms
from obsarray.templater.template_util import create_ds
from obsarray.templater.dstemplater import DSTemplater
from obsarray.templater.dswriter import DSWriter
from obsarray.utils import append_names
from obsarray.options import set_options, get_options
from obsarray.conventions import UNCConventionsWarning

try:
    __version__ = version("obsarray")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
