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
        self._sli = self.expand_sli(sli)

        return self

    def __setitem__(self, sli: tuple, flag_value):
        """
        Sets flag values

        :param sli: slice of variable
        :param flag_value: flag value as a single value boolean to apply to all data or boolean mask array
        """

        self._sli = self.expand_sli(sli)

        flag_meanings, flag_masks = DatasetUtil.unpack_flag_attrs(
            self._obj[self._flag_var_name].attrs
        )

        flag_bit = flag_meanings.index(self._flag_meaning)
        flag_mask = flag_masks[flag_bit]

        # if boolean to apply to all data
        if isinstance(flag_value, bool):
            if flag_value:
                self._obj[self._flag_var_name][self._sli].values[:] = (
                    self._obj[self._flag_var_name][self._sli].values | flag_mask
                )
            else:
                self._obj[self._flag_var_name][self._sli].values[:] = (
                    self._obj[self._flag_var_name][self._sli].values & ~flag_mask
                )

        # else apply mask
        else:
            if flag_value.dtype is not np.dtype(bool):
                TypeError("Flag mask must of boolean type")

            i_true = np.where(flag_value == True)
            i_false = np.where(flag_value == False)

            self._obj[self._flag_var_name][self._sli].values[i_true] = (
                self._obj[self._flag_var_name][self._sli].values[i_true] | flag_mask
            )

            self._obj[self._flag_var_name][self._sli].values[i_false] = (
                self._obj[self._flag_var_name][self._sli].values[i_false] & ~flag_mask
            )

    def expand_sli(self, sli: Optional[tuple] = None) -> tuple:
        """
        Function to expand the provided sli so that it always has the right number of dimensions

        :param sli: input sli tuple. This one can have fewer dimensions than the total if e.g. only providing the first index
        :return: output sli tuple
        """

        # if no slice provided, define as slice for full array
        if sli is None:
            out_sli = tuple([slice(None)] * self._obj[self._flag_var_name].ndim)

        else:
            out_sli = list([slice(None)] * self._obj[self._flag_var_name].ndim)
            sli_list = list(sli)
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


class DatasetUtil2:
    @staticmethod
    def create_flags_variable(dim_sizes, meanings, dim_names=None, attributes=None):
        """
        Return default empty 1d xarray flag Variable

        :type dim_sizes: list
        :param dim_sizes: dimension sizes as ints, i.e. [dim1_size, dim2_size, dim3_size] (e.g. [2,3,5])

        :type attributes: dict
        :param attributes: (optional) dictionary of variable attributes, e.g. standard_name

        :type dim_names: list
        :param dim_names: (optional) dimension names as strings, i.e. ["dim1_name", "dim2_name", "dim3_size"]

        :return: Default empty flag vector variable
        :rtype: xarray.Variable
        """

        n_masks = len(meanings)

        data_type = DatasetUtil.return_flags_dtype(n_masks)

        variable = DatasetUtil.create_variable(
            dim_sizes,
            data_type,
            dim_names=dim_names,
            attributes=attributes,
        )

        #initialise flags to zero (instead of fillvalue)
        variable.values=0*variable.values

        # add flag attributes
        variable.attrs["flag_meanings"] = (
            str(meanings)[1:-1].replace("'", "").replace(",", "")
        )
        variable.attrs["flag_masks"] = str([2 ** i for i in range(0, n_masks)])[1:-1]

        # todo - make sure flags can't have units

        return variable

    @staticmethod
    def return_flags_dtype(n_masks):
        """
        Return required flags array data type

        :type n_masks: int
        :param n_masks: number of masks required in flag array

        :return: data type
        :rtype: dtype
        """

        if n_masks <= 8:
            return np.int8
        elif n_masks <= 16:
            return np.int16
        elif n_masks <= 32:
            return np.int32
        else:
            return np.int64

    @staticmethod
    def add_encoding(
        variable, dtype, scale_factor=1.0, offset=0.0, fill_value=None, chunksizes=None
    ):
        """
        Add encoding to xarray Variable to apply when writing netCDF files

        :type variable: xarray.Variable
        :param variable: data variable

        :type dtype: type
        :param dtype: numpy data type

        :type scale_factor: float
        :param scale_factor: variable scale factor

        :type offset: float
        :param offset: variable offset value

        :type fill_value: int/float
        :param fill_value: (optional) fill value

        :type chunksizes: float
        :param chunksizes: (optional) chucksizes
        """

        # todo - make sure flags can't have encoding added

        encoding_dict = {
            "dtype": dtype,
            "scale_factor": scale_factor,
            "add_offset": offset,
        }

        if chunksizes is not None:
            encoding_dict.update({"chunksizes": chunksizes})

        if fill_value is not None:
            encoding_dict.update({"_FillValue": fill_value})

        variable.encoding = encoding_dict

    @staticmethod
    def get_default_fill_value(dtype):
        """
        Returns default fill_value for given data type

        :type dtype: type
        :param dtype: numpy dtype

        :return: CF-conforming fill value
        :rtype: fill_value
        """

        if dtype == np.int8:
            return np.int8(-129)
        if dtype == np.uint8:
            return np.uint8(-1)
        elif dtype == np.int16:
            return np.int16(-32769)
        elif dtype == np.uint16:
            return np.uint16(-1)
        elif dtype == np.int32:
            return np.int32(-2147483649)
        elif dtype == np.uint32:
            return np.uint32(-1)
        elif dtype == np.int64:
            return np.int64(-9223372036854775808)
        elif dtype == np.float32:
            return np.float32(9.96921e36)
        elif dtype == np.float64:
            return np.float64(9.969209968386869e36)

    @staticmethod
    def _get_flag_encoding(da):
        """
        Returns flag encoding for flag type data array
        :type da: xarray.DataArray
        :param da: data array
        :return: flag meanings
        :rtype: list
        :return: flag masks
        :rtype: list
        """

        try:
            flag_meanings = da.attrs["flag_meanings"].split()
            flag_masks = [int(fm) for fm in da.attrs["flag_masks"].split(",")]
        except KeyError:
            raise KeyError(da.name + " not a flag variable")

        return flag_meanings, flag_masks

    @staticmethod
    def unpack_flags(da):
        """
        Breaks down flag data array into dataset of boolean masks for each flag
        :type da: xarray.DataArray
        :param da: dataset
        :return: flag masks
        :rtype: xarray.Dataset
        """

        flag_meanings, flag_masks = DatasetUtil._get_flag_encoding(da)

        ds = Dataset()
        for flag_meaning, flag_mask in zip(flag_meanings, flag_masks):
            ds[flag_meaning] = DatasetUtil.create_variable(
                list(da.shape), bool, dim_names=list(da.dims)
            )
            ds[flag_meaning] = (da & flag_mask).astype(bool)

        return ds

    @staticmethod
    def get_flags_mask_or(da, flags=None):
        """
        Returns boolean mask for set of flags, defined as logical or of flags

        :type da: xarray.DataArray
        :param da: dataset

        :type flags: list
        :param flags: list of flags (if unset all data flags selected)

        :return: flag masks
        :rtype: numpy.ndarray
        """

        flags_ds = DatasetUtil.unpack_flags(da)

        flags = flags if flags is not None else flags_ds.variables
        mask_flags = [flags_ds[flag].values for flag in flags]

        return np.logical_or.reduce(mask_flags)

    @staticmethod
    def get_flags_mask_and(da, flags=None):
        """
        Returns boolean mask for set of flags, defined as logical and of flags

        :type da: xarray.DataArray
        :param da: dataset

        :type flags: list
        :param flags: list of flags (if unset all data flags selected)

        :return: flag masks
        :rtype: numpy.ndarray
        """

        flags_ds = DatasetUtil.unpack_flags(da)

        flags = flags if flags is not None else flags_ds.variables
        mask_flags = [flags_ds[flag].values for flag in flags]

        return np.logical_and.reduce(mask_flags)

    @staticmethod
    def set_flag(da, flag_name, error_if_set=False):
        """
        Sets named flag for elements in data array
        :type da: xarray.DataArray
        :param da: dataset
        :type flag_name: str
        :param flag_name: name of flag to set
        :type error_if_set: bool
        :param error_if_set: raises error if chosen flag is already set for any element
        """

        set_flags = DatasetUtil.unpack_flags(da)[flag_name]

        if np.any(set_flags == True) and error_if_set:
            raise ValueError(
                "Flag " + flag_name + " already set for variable " + da.name
            )

        # Find flag mask
        flag_meanings, flag_masks = DatasetUtil._get_flag_encoding(da)
        flag_bit = flag_meanings.index(flag_name)
        flag_mask = flag_masks[flag_bit]

        da.values = da.values | flag_mask

        return da

    @staticmethod
    def unset_flag(da, flag_name, error_if_unset=False):
        """
        Unsets named flag for specified index of dataset variable
        :type da: xarray.DataArray
        :param da: data array
        :type flag_name: str
        :param flag_name: name of flag to unset
        :type error_if_unset: bool
        :param error_if_unset: raises error if chosen flag is already set at specified index
        """

        set_flags = DatasetUtil.unpack_flags(da)[flag_name]

        if np.any(set_flags == False) and error_if_unset:
            raise ValueError(
                "Flag " + flag_name + " already set for variable " + da.name
            )

        # Find flag mask
        flag_meanings, flag_masks = DatasetUtil._get_flag_encoding(da)
        flag_bit = flag_meanings.index(flag_name)
        flag_mask = flag_masks[flag_bit]

        da.values = da.values & ~flag_mask

        return da

    @staticmethod
    def get_set_flags(da):
        """
        Return list of set flags for single element data array
        :type da: xarray.DataArray
        :param da: single element data array
        :return: set flags
        :rtype: list
        """

        if da.shape != ():
            raise ValueError("Must pass single element data array")

        flag_meanings, flag_masks = DatasetUtil._get_flag_encoding(da)

        set_flags = []
        for flag_meaning, flag_mask in zip(flag_meanings, flag_masks):
            if da & flag_mask:
                set_flags.append(flag_meaning)

        return set_flags

    @staticmethod
    def check_flag_set(da, flag_name):
        """
        Returns if flag for single element data array
        :type da: xarray.DataArray
        :param da: single element data array
        :type flag_name: str
        :param flag_name: name of flag to set
        :return: set flags
        :rtype: list
        """

        if da.shape != ():
            raise ValueError("Must pass single element data array")

        set_flags = DatasetUtil.get_set_flags(da)

        if flag_name in set_flags:
            return True
        return False
