"""
Utilities for creating xarray dataset variables in specified forms
"""

import string
import xarray
import numpy
from typing import Union, Optional, List, Dict


__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"


DEFAULT_DIM_NAMES = list(string.ascii_lowercase[-3:]) + list(
    string.ascii_lowercase[:-3]
)
DEFAULT_DIM_NAMES.reverse()


ERR_CORR_DEFS = {
    "random": {"n_params": 0},
    "rectangle_absolute": {"n_params": 2},
    "rectangular_relative": {"n_params": 2},
    "triangular_relative": {"n_params": 2},
}


class DatasetUtil:
    """
    Class to provide utilities for generating standard xarray DataArrays and Variables
    """

    @staticmethod
    def create_default_array(
        dim_sizes: List[int],
        dtype: numpy.typecodes,
        dim_names: Optional[List[str]] = None,
        fill_value: Optional[Union[int, float]] = None,
    ) -> xarray.DataArray:
        """
        Return default empty xarray DataArray

        :param dim_sizes: dimension sizes, i.e. ``[dim1_size, dim2_size, dim3_size]`` (e.g. ``[2,3,5]``)
        :param dtype: numpy data type
        :param dim_names: dimension names, i.e. ``["dim1_name", "dim2_name", "dim3_name"]``
        :param fill_value: fill value (if None CF compliant value used)

        :returns: Default empty array
        """

        if fill_value is None:
            fill_value = DatasetUtil.get_default_fill_value(dtype)

        empty_array = numpy.full(dim_sizes, fill_value, dtype)

        if dim_names is not None:
            default_array = xarray.DataArray(empty_array, dims=dim_names)
        else:
            default_array = xarray.DataArray(
                empty_array, dims=DEFAULT_DIM_NAMES[-len(dim_sizes) :]
            )

        return default_array

    @staticmethod
    def create_variable(
        dim_sizes: List[int],
        dtype: numpy.typecodes,
        dim_names: Optional[List[str]] = None,
        attributes: Dict = None,
        fill_value: Optional[Union[int, float]] = None,
    ) -> xarray.Variable:
        """
        Return default empty xarray Variable

        :param dim_sizes: dimension sizes, i.e. ``[dim1_size, dim2_size, dim3_size]`` (e.g. ``[2,3,5]``)
        :param dtype: numpy data type
        :param dim_names: dimension names as strings, i.e. ``["dim1_name", "dim2_name", "dim3_size"]``
        :param attributes: dictionary of variable attributes, e.g. standard_name
        :param fill_value: fill value (if None CF compliant value used)

        :returns: Default empty variable
        """

        if fill_value is None:
            fill_value = DatasetUtil.get_default_fill_value(dtype)

        default_array = DatasetUtil.create_default_array(
            dim_sizes, dtype, fill_value=fill_value
        )

        if dim_names is None:
            variable = xarray.Variable(
                DEFAULT_DIM_NAMES[-len(dim_sizes) :], default_array
            )
        else:
            variable = xarray.Variable(dim_names, default_array)

        variable.attrs["_FillValue"] = fill_value

        if attributes is not None:
            variable.attrs = {**variable.attrs, **attributes}

        return variable

    @staticmethod
    def create_unc_variable(
        dim_sizes: List[int],
        dtype: numpy.typecodes,
        dim_names: List[str],
        attributes: Optional[Dict] = None,
        pdf_shape: str = "gaussian",
        err_corr: Optional[List[Dict[str, Union[str, List]]]] = None,
    ) -> xarray.Variable:
        """
        Return default empty 1d xarray uncertainty Variable

        :param dim_sizes: dimension sizes, i.e. ``[dim1_size, dim2_size, dim3_size]`` (e.g. ``[2,3,5]``)
        :param dtype: data type
        :param dim_names: dimension names, i.e. ``["dim1_name", "dim2_name", "dim3_size"]``
        :param attributes: dictionary of variable attributes, e.g. standard_name
        :param pdf_shape: (default: `"gaussian"`) pdf shape of uncertainties
        :param err_corr: uncertainty error-correlation structure definition, defined as below.

        :returns: Default empty flag vector variable

        Each element of ``err_corr``  is a dictionary that defines the error-correlation along one or more dimensions,
        which should include the following entries:

        * ``dim`` (*str*/*list*) - name of the dimension(s) as a ``str`` or list of ``str``'s (i.e. from ``dim_names``)
        * ``form`` (*str*) - error-correlation form name, functional form of error-correlation structure for
          dimension(s)
        * ``params`` (*list*) - (optional) parameters of the error-correlation structure defining function for dimension
          if required. The number of parameters required depends on the particular form.
        * ``units`` (*list*) - (optional) units of the error-correlation function parameters for dimension
          (ordered as the parameters)

        For more information on the required form of these entries, see the :ref:`uncertainties section <err corr>` of
        the user guide.

        .. note::
            If the error-correlation structure is not defined along a particular dimension (i.e. it is not
            included in ``err_corr``), the error-correlation is assumed random. Variable attributes are
            populated to the effect of this assumption.
        """

        # define uncertainty variable attributes, based on FIDUCEO Full FCDR definition (if required)
        attributes = {} if attributes is None else attributes

        if err_corr is None:
            err_corr = []

        # set undefined dims as random
        defined_err_corr_dims = []
        for erd in err_corr:
            defined_err_corr_dims.append(erd["dim"])

        missing_err_corr_dims = [
            dim for dim in dim_names if dim not in defined_err_corr_dims
        ]
        for missing_err_corr_dim in missing_err_corr_dims:
            err_corr.append({"dim": missing_err_corr_dim, "form": "random"})

        for i, ecdef in enumerate(err_corr):
            idx = str(i + 1)

            dim_str = DatasetUtil.return_err_corr_dim_str(idx)
            form_str = DatasetUtil.return_err_corr_form_str(idx)
            params_str = DatasetUtil.return_err_corr_params_str(idx)
            units_str = DatasetUtil.return_err_corr_units_str(idx)

            form = ecdef["form"]
            attributes[dim_str] = ecdef["dim"]
            attributes[form_str] = ecdef["form"]
            attributes[units_str] = ecdef["units"] if "units" in ecdef else []

            # if defined form, check number of params valid
            if "params" in ecdef:
                if form in ERR_CORR_DEFS.keys():
                    n_params = len(ecdef["params"])
                    req_n_params = ERR_CORR_DEFS[form]["n_params"]
                    if n_params != req_n_params:
                        raise ValueError(
                            "Must define "
                            + str(req_n_params)
                            + " for correlation form"
                            + form
                            + "(not "
                            + str(n_params)
                            + ")"
                        )

                attributes[params_str] = ecdef["params"]

            else:
                attributes[params_str] = []

        attributes["pdf_shape"] = pdf_shape

        # Create variable
        variable = DatasetUtil.create_variable(
            dim_sizes,
            dtype,
            dim_names=dim_names,
            attributes=attributes,
        )

        return variable

    @staticmethod
    def return_err_corr_dim_str(idx):
        return "_".join(["err", "corr", idx, "dim"])

    @staticmethod
    def return_err_corr_form_str(idx):
        return "_".join(["err", "corr", idx, "form"])

    @staticmethod
    def return_err_corr_params_str(idx):
        return "_".join(["err", "corr", idx, "params"])

    @staticmethod
    def return_err_corr_units_str(idx):
        return "_".join(["err", "corr", idx, "units"])

    @staticmethod
    def create_flags_variable(
        dim_sizes: List[int],
        meanings: List[str],
        dim_names: Optional[List[str]] = None,
        attributes: Optional[Dict] = None,
    ) -> xarray.Variable:
        """
        Return default empty 1d xarray flag Variable

        :param dim_sizes: dimension sizes, i.e. ``[dim1_size, dim2_size, dim3_size]`` (e.g. ``[2,3,5]``)
        :param meanings: flag meanings by bit
        :param dim_names: dimension names, i.e. ``["dim1_name", "dim2_name", "dim3_size"]``
        :param attributes: dictionary of variable attributes, e.g. standard_name

        :returns: Default empty flag vector variable
        """

        n_masks = len(meanings)

        data_type = DatasetUtil.return_flags_dtype(n_masks)

        variable = DatasetUtil.create_variable(
            dim_sizes,
            data_type,
            dim_names=dim_names,
            fill_value=0,
            attributes=attributes,
        )

        # add flag attributes
        variable.attrs["flag_meanings"] = (
            str(meanings)[1:-1].replace("'", "").replace(",", "")
        )
        variable.attrs["flag_masks"] = str([2 ** i for i in range(0, n_masks)])[1:-1]

        # todo - make sure flags can't have units

        return variable

    @staticmethod
    def return_flags_dtype(n_masks: int) -> numpy.typecodes:
        """
        Return required flags array data type

        :param n_masks: number of masks required in flag array
        :return: data type
        """

        if n_masks <= 8:
            return numpy.uint8
        elif n_masks <= 16:
            return numpy.uint16
        elif n_masks <= 32:
            return numpy.uint32
        else:
            return numpy.uint64

    @staticmethod
    def add_encoding(
        variable: xarray.Variable,
        dtype: numpy.typecodes,
        scale_factor: Optional[float] = 1.0,
        offset: Optional[float] = 0.0,
        fill_value: Optional[Union[int, float]] = None,
        chunksizes: Optional[float] = None,
    ):
        """
        Add encoding to xarray Variable to apply when writing netCDF files

        :param variable: data variable
        :param dtype: numpy data type
        :param scale_factor: variable scale factor
        :param offset: variable offset value
        :param fill_value: fill value
        :param chunksizes: chucksizes
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
    def get_default_fill_value(dtype: numpy.typecodes) -> Union[int, float]:
        """
        Returns default fill_value for given data type

        :param dtype: numpy data type

        :return: CF-conforming fill value
        """

        if dtype == numpy.int8:
            return numpy.int8(-127)
        if dtype == numpy.uint8:
            return numpy.uint8(-1)
        elif dtype == numpy.int16:
            return numpy.int16(-32767)
        elif dtype == numpy.uint16:
            return numpy.uint16(-1)
        elif dtype == numpy.int32:
            return numpy.int32(-2147483647)
        elif dtype == numpy.uint32:
            return numpy.uint32(-1)
        elif dtype == numpy.int64:
            return numpy.int64(-9223372036854775806)
        elif dtype == numpy.float32:
            return numpy.float32(9.96921e36)
        elif dtype == numpy.float64:
            return numpy.float64(9.969209968386869e36)


if __name__ == "__main__":
    pass
