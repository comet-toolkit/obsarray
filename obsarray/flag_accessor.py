"""unc_accessor - xarray extensions with accessor objects for uncertainty handling"""

import xarray as xr


__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"
__all__ = []


@xr.register_dataset_accessor("flag")
class FlagAccessor(object):
    """
    ``xarray.Dataset`` accesssor object for handling dataset variable flags

    :param xarray_obj: xarray dataset
    """

    def __init__(self, xarray_obj: xr.Dataset):

        # Initialise attributes
        self._obj = xarray_obj


if __name__ == "__main__":
    pass
