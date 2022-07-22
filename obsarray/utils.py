"""utils - utilities for obsarray"""

import numpy as np
import xarray as xr


__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"
__all__ = ["empty_err_corr_matrix"]


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


if __name__ == "__main__":
    pass
