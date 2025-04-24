"""obsarray.concat - module with extension to xarray.concat for obs_vars and unc_vars"""

import numpy as np
import xarray as xr
from typing import Union, Any
from xarray.core.types import T_Dataset, T_DataArray, T_Variable
from collections.abc import Hashable, Iterable


__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"
__all__ = []


def obs_concat(
    objs: Iterable[T_DataArray],
    dim: Union[Hashable, T_Variable, T_DataArray, Any],
    unc: T_Dataset,
    dim_err_corr: Iterable[Any],
    combine_unc: str = "concat",
    *args,
    **kwargs
):
    """
    Concatenate xarray *obs_vars* along a new or existing dimension, safely handling also
    concatenating associated *unc_vars*. Extension to :py:func:`xarray.concat`.

    :param objs: sequence of :py:class:`xarray.Dataset` and :py:class:`xarray.DataArray`
        xarray objects to concatenate *obs_vars* together. As for :py:class:`xarray.Dataset`,
        each object is expected to consist of variables and coordinates with matching shapes
        except for along the concatenated dimension.
    :param dim: Name of the dimension to concatenate along. This can either be a new
        dimension name, in which case it is added along axis=0, or an existing
        dimension name, in which case the location of the dimension is
        unchanged. If dimension is provided as a Variable, DataArray or Index, its name
        is used as the dimension to concatenate along and the values are added
        as a coordinate.
    :param unc: dataset containing the unc_vars associated with objs
    :param dim_err_corr: error-correlation form definition for concatenation dimension
    :param combine_unc: string indicating how to concatenate unc_vars.

        * "concat": (default) merges *unc_vars* as for *obs_vars* - assumes *unc_var* order is the same between *obs_vars*
        * "no_concat": expands each *unc_var* along dim, gap filling with zeros
    """

    concat_obs_vars = xr.concat(objs, dim, *args, **kwargs)
    concat_unc_vars = xr.Dataset

    n_uc = np.array([len(obj.attrs["unc_comps"]) for obj in objs])

    if np.all(n_uc != n_uc[0]):
        raise ValueError()
    else:
        n_uc = n_uc[0]

    for i_uc in range(n_uc):
        unc_i = [unc[obj.attrs["unc_comps"][i_uc]] for obj in objs]


    return concat_obs_vars, concat_unc_vars


if __name__ == "__main__":
    pass
