"""flag_accessor - xarray extensions with accessor objects for flag handling"""

import numpy as np
import xarray as xr
from typing import List, Optional, Union, Tuple
from obsarray.templater.template_util import create_var
from obsarray.templater.dataset_util import DatasetUtil


__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"
__all__ = []


class Flag:
    """
    Interface for handling ``xarray.Dataset`` flag variable flags

    :param xarray_obj: dataset
    :param flag_var_name: name of flag variable
    :param flag_meaning: specific flag from flag variable
    """

    def __init__(
        self,
        xarray_obj: xr.Dataset,
        flag_var_name: str,
        flag_meaning: str,
    ):

        # initialise attributes

        self._obj = xarray_obj
        self._flag_var_name = flag_var_name
        self._flag_meaning = flag_meaning
        self._sli = tuple([slice(None)] * self._obj[self._flag_var_name].ndim)

    def __str__(self):
        """Custom __str__"""
        return "<{}> \n{}".format(
            self.__class__.__name__,
            self.value.__repr__()[18:],
        )

    def __repr__(self):
        """Custom  __repr__"""
        return str(self)

    def __getitem__(self, sli: tuple):
        """
        Defines variable slice

        :param sli: slice of variable
        :return: self
        """

        # update slice
        self._sli = self._expand_sli(sli)

        return self

    def __setitem__(self, sli: tuple, flag_value):
        """
        Sets flag values

        :param sli: slice of variable
        :param flag_value: flag value as a single value boolean to apply to all data or boolean mask array
        """

        self._sli = self._expand_sli(sli)

        flag_meanings, flag_masks = DatasetUtil.unpack_flag_attrs(
            self._obj[self._flag_var_name].attrs
        )

        flag_bit = flag_meanings.index(self._flag_meaning)
        flag_mask = flag_masks[flag_bit]

        # if boolean to apply to all data
        if isinstance(flag_value, bool):
            if flag_value:
                self._obj[self._flag_var_name].values[self._sli] = (
                    self._obj[self._flag_var_name][self._sli].values | flag_mask
                )
            else:
                self._obj[self._flag_var_name].values[self._sli] = (
                    self._obj[self._flag_var_name][self._sli].values & ~flag_mask
                )

        # else apply mask
        else:
            if flag_value.dtype is not np.dtype(bool):
                TypeError("Flag mask must of boolean type")

            i_true = np.where(flag_value == True)
            i_false = np.where(flag_value == False)

            self._obj[self._flag_var_name][self._sli].values[i_true] = np.array(
                self._obj[self._flag_var_name][self._sli].values[i_true] | flag_mask
            )

            self._obj[self._flag_var_name][self._sli].values[i_false] = np.array(
                self._obj[self._flag_var_name][self._sli].values[i_false] & ~flag_mask
            )

    def _expand_sli(self, sli: Optional[tuple] = None) -> tuple:
        """
        Function to expand the provided sli so that it always has the right number of dimensions

        :param sli: input sli tuple. This one can have fewer dimensions than the total if e.g. only providing the first index
        :return: output sli tuple
        """

        # if no slice provided, define as slice for full array
        if sli is None:
            out_sli = tuple([slice(None)] * self._obj[self._flag_var_name].ndim)

        # Otherwise, set each : dimension to slice(None)
        # E.g. if providing [0] for a variable with 3 dimensions, this becomes
        # [0,slice(None),slice(None)]
        else:
            out_sli = list([slice(None)] * self._obj[self._flag_var_name].ndim)
            sli_list = list(sli) if isinstance(sli, tuple) else [sli]
            for i in range(len(sli_list)):
                if not sli_list[i] == ":":
                    out_sli[i] = sli_list[i]
            out_sli = tuple(out_sli)

        return out_sli

    @property
    def value(self) -> xr.DataArray:
        """
        Return flag variable flag value

        :return: flag variable flag value
        """

        flag_meanings, flag_masks = DatasetUtil.unpack_flag_attrs(
            self._obj[self._flag_var_name].attrs
        )

        flag_bit = flag_meanings.index(self._flag_meaning)
        flag_mask = flag_masks[flag_bit]

        value = xr.DataArray(
            np.zeros(self._obj[self._flag_var_name][self._sli].shape, dtype=bool),
            dims=list(self._obj[self._flag_var_name][self._sli].dims),
        )

        value.values[:] = (
            self._obj[self._flag_var_name][self._sli] & flag_mask
        ).astype(bool)

        return value


class FlagVariable:
    """
    Interface for handling ``xarray.Dataset`` flag variables

    :param xarray_obj: dataset
    :param flag_var_name: name of flag variable
    """

    def __init__(self, xarray_obj: xr.Dataset, flag_var_name: str):

        # Initialise attributes
        self._obj = xarray_obj
        self._flag_var_name = flag_var_name

    def __getitem__(self, key: str) -> "Flag":
        """
        Returns flag variable flag interface

        :param key: flag meaning
        :return: flag interface
        """

        return Flag(self._obj, self._flag_var_name, key)

    def __setitem__(self, flag_meaning: str, flag_value: Union[bool, np.ndarray]):
        """
        Sets defined flag variable flag, if flag is:

        * an existing flag (i.e., in variable ``flag_meanings`` list), sets data for that flag
        * not an existing flag variable, that flag is added to ``flag_meanings`` and has its data set. Note this fails if max number of flags for flag variable dtype already defined (e.g. 8 flags defined for ``int8`` variable)

        :param flag_meaning: flag meaning name
        :param flag_value: flag value as a single value boolean to apply to all data or boolean mask array
        """

        flag_meanings, flag_masks = DatasetUtil.unpack_flag_attrs(
            self._obj[self._flag_var_name].attrs
        )

        if flag_meaning not in flag_meanings:
            self._obj[
                self._flag_var_name
            ].attrs = DatasetUtil.add_flag_meaning_to_attrs(
                self._obj[self._flag_var_name].attrs,
                flag_meaning,
                self._obj[self._flag_var_name].dtype,
            )

        self[flag_meaning][:] = flag_value

    def __delitem__(self, flag_meaning):
        """
        Removes defined flag variable flag

        :param flag_meaning: flag meaning name
        """

        self[flag_meaning][:] = False

        self._obj[self._flag_var_name].attrs = DatasetUtil.rm_flag_meaning_from_attrs(
            self._obj[self._flag_var_name].attrs,
            flag_meaning,
        )

    def __str__(self):
        """Custom __str__"""
        return "<{}>\nFlagVariable: '{}'\n{}".format(
            self.__class__.__name__,
            self._flag_var_name,
            self.keys().__repr__(),
        )

    def __repr__(self):
        """Custom  __repr__"""
        return str(self)

    def __len__(self) -> int:
        """
        Returns number of flag variable flags

        :returns: number of flag variable flags
        """
        return len(self.keys())

    def __iter__(self):
        """Custom  __iter__"""

        self.i = 0  # Define counter
        return self

    def __next__(self) -> Flag:
        """
        Returns ith flag in flag variable

        :return: flag variable flag
        """

        # Iterate through flag variable flags
        if self.i < len(self.keys()):
            self.i += 1  # Update counter
            return self[self.keys()[self.i - 1]]

        else:
            raise StopIteration

    def keys(self) -> List[str]:
        """
        Returns flag variable flag names

        :return: flag variable flag names
        """
        flag_meanings, _ = DatasetUtil.unpack_flag_attrs(
            self._obj[self._flag_var_name].attrs
        )

        return flag_meanings

    # todo - add return set flags - array of objects, where object is list of set flags


@xr.register_dataset_accessor("flag")
class FlagAccessor(object):
    """
    ``xarray.Dataset`` accesssor object for handling dataset variable flags

    :param xarray_obj: xarray dataset
    """

    def __init__(self, xarray_obj: xr.Dataset):

        # Initialise attributes
        self._obj = xarray_obj

    def __str__(self) -> str:
        """Custom __str__"""

        string = "<{}>\nDataset Flags:\n".format(self.__class__.__name__)

        for var_name in self.keys():
            string += "* {}\n".format(self[var_name].__repr__())

        return string

    def __repr__(self) -> str:
        """Custom  __repr__"""
        return str(self)

    def __getitem__(self, flag_var_name: str) -> FlagVariable:
        """Custom  __repr__"""
        return FlagVariable(self._obj, flag_var_name)

    def __setitem__(
        self,
        flag_var_name: str,
        flag_def: Union[xr.DataArray, xr.Variable, Tuple[List[str], dict]],
    ):
        """
        Adds defined flag variable to dataset

        :param flag_var_name: flag variable name
        :param flag_def: either xarray DataArray/Variable, or definition through tuple as ``(dims, attrs)``. ``dims`` is a list of variable dimension names, and ``attrs`` is a dictionary of variable attributes. ``attrs`` should include an element ``flag_meanings`` which is a list defining the flag variable flags.
        """

        # add uncertainty variable
        if type(flag_def) == xr.DataArray:

            # add necessary flag metadata if missing from provided data array
            flag_meanings, flag_masks = DatasetUtil.unpack_flag_attrs(flag_def.attrs)
            flag_def.attrs.update(DatasetUtil.pack_flag_attrs(flag_meanings))

            self._obj[flag_var_name] = flag_def

        # use templater functionality if var defined with tuple
        elif type(flag_def) == tuple:

            attrs = flag_def[1]

            if "flag_meanings" not in attrs:
                attrs["flag_meanings"] = []

            var_attrs = {
                "dtype": "flag",
                "dim": flag_def[0],
                "attributes": attrs,
            }

            size = {d: len(self._obj[d]) for d in flag_def[0]}

            var = create_var(flag_var_name, var_attrs, size)

            self._obj[flag_var_name] = var

    def __len__(self):
        """
        Returns number of flag variables

        :return: number of flag variables
        """

        return len(self.keys())

    def __iter__(self) -> "FlagAccessor":
        """
        Initialises iterator

        :return: self
        """
        self.i = 0  # Define counter
        return self

    def __next__(self):
        """
        Returns ith flag variable

        :return: ith flag variable
        """

        # Iterate through flag variables
        if self.i < len(self.keys()):
            self.i += 1  # Update counter
            return self[self.keys()[self.i - 1]]

        else:
            raise StopIteration

    def keys(self):
        """
        Returns flag variable names

        :return: flag variable names
        """

        return list(self._obj.flag.flag_vars.keys())

    @property
    def data_vars(self) -> xr.core.dataset.DataVariables:
        """
        Returns dataset data variables (defined as dataset variables that are not themselves flags)

        :return: data variables
        """
        data_var_names = []
        for var_name in self._obj.variables:
            if self._is_data_var(var_name):
                data_var_names.append(var_name)
        return self._obj[data_var_names].data_vars

    @property
    def flag_vars(self) -> xr.core.dataset.DataVariables:
        """
        Returns dataset flag variables (defined as flags associated with data variables)

        :return: flag variables
        """
        flag_var_names = []
        for var_name in self._obj.variables:
            if self._is_flag_var(var_name):
                flag_var_names.append(var_name)

        return self._obj[list(flag_var_names)].data_vars

    def _is_data_var(self, var_name: str) -> bool:
        """
        Returns true if named dataset variable is an data variable (i.e. not a flag variable)

        :return: data variable bool
        """

        if not self._is_flag_var(var_name):
            return True
        return False

    def _is_flag_var(self, var_name: str) -> bool:
        """
        Returns true if named dataset variable is an flag variable

        :return: flag variable bool
        """

        if "flag_meanings" in self._obj[var_name].attrs:
            return True
        return False


if __name__ == "__main__":
    pass
