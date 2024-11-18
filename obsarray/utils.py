"""utils - utilities for obsarray"""

import numpy as np
import xarray as xr
from xarray.core.types import T_Dataset

__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"
__all__ = ["empty_err_corr_matrix", "append_names"]


def empty_err_corr_matrix(obs_var: xr.DataArray):
    """
    Returns diagonal error correlation matrix for observation variable

    :param obs_var: uncertainty variable
    :return: diagonal error correlation matrix
    """

    dim_names = [dim for dim in obs_var.dims]
    dim_lens = [len(obs_var[dim]) for dim in dim_names]

    n_elems = int(np.prod(dim_lens))
    erc_dim_name = ".".join(dim_names)

    err_corr_matrix = xr.DataArray(np.eye(n_elems), dims=[erc_dim_name, erc_dim_name])

    return err_corr_matrix


def append_names(
    ds: T_Dataset,
    suffix: str,
    skip_vars: bool = False,
    skip_dims: bool = False,
    skip_attrs: bool = False,
) -> T_Dataset:
    """
    Appends a suffix to the names of dataset variables, dimensions, and attributes - safely handling `unc_vars` and associated metadata

    :param ds: xarray dataset
    :param suffix: suffix to append to dataset variable, dimension, and attribute names
    :param skip_vars: (default: `False`) switch to skip applying suffix to variable names
    :param skip_dims: (default: `False`) switch to skip applying suffix to dimension names
    :param skip_attrs: (default: `False`) switch to skip applying suffix to variable names
    :returns: ds with suffix appended to names of variables, dimensions, attributes
    """

    # update variable names
    if not skip_vars:
        var_rename = {var_name: var_name + suffix for var_name in ds.variables.keys()}
        ds = ds.unc.rename(var_rename)

    # update dimension names
    if not skip_dims:
        dim_rename = {dim_name: dim_name + suffix for dim_name in ds.dims.keys()}
        ds = ds.unc.rename_dims(dim_rename)

    # update attribute names
    if not skip_attrs:
        ds.attrs = {key + suffix: value for key, value in ds.attrs.items()}

    return ds


if __name__ == "__main__":
    pass
