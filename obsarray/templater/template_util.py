"""
Utilities for creating template xarray datasets
"""

from typing import Optional, Dict, List
from obsarray.templater.dataset_util import DatasetUtil
import xarray


__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"


def create_ds(
    template: Dict[str, Dict], size: Dict[str, int], metadata: Optional[Dict] = None
) -> xarray.Dataset:
    """
    Returns template dataset

    :param template: dictionary defining ds variable structure, as defined below.
    :param size: dictionary of dataset dimensions, entry per dataset dimension with value of size as int
    :param metadata: dictionary of dataset metadata

    :returns: template dataset

    For the ``template`` dictionary each key/value pair defines one variable, where the key is the variable name and the value is a dictionary with the following entries:

    * ``"dtype"`` (*np.typecodes*/*str*) - variable data type, either a numpy data type or special value ``"flag"`` for
      flag variable
    * ``"dim"`` (*list*) - list of variable dimension names
    * ``"attributes"`` (*dict*) - (optional) variable attributes
    * ``"encoding"`` (*dict*) - (optional) variable encoding.

    For more information on the required form of these entries, see the :ref:`variables definition section <variables dict>`
    of the user guide.
    """

    # Create dataset
    ds = xarray.Dataset()

    # Add variables
    ds = TemplateUtil.add_variables(ds, template, size)

    # Add metadata
    if metadata is not None:
        ds = TemplateUtil.add_metadata(ds, metadata)

    return ds


def create_var(var_name: str, var_attrs: Dict, size: Dict[str, int]) -> xarray.Variable:
    """
    Returns template variable

    :param var_name: variable name
    :param var_attrs: variable definition dictionary (as an entry to a template dictionary)
    :param size: dictionary of dataset dimensions, entry per dataset dimension with value of size as int
    :return:
    """

    return TemplateUtil._create_var(var_name, var_attrs, size)


class TemplateUtil:
    """
    Class to create template xarray datasets
    """

    @staticmethod
    def add_variables(
        ds: xarray.Dataset, template: Dict[str, Dict], size: Dict[str, int]
    ) -> xarray.Dataset:
        """
        Adds defined variables dataset

        :param ds: dataset
        :param template: dictionary defining variables, see the :ref:`variables definition section <variables dict>` of the user guide for more information.
        :param size: dictionary of dataset dimensions, entry per dataset dimension with value of size as int

        :returns: dataset with defined variables
        """

        for var_name in template.keys():
            var = TemplateUtil._create_var(var_name, template[var_name], size)

            ds[var_name] = var

        return ds

    @staticmethod
    def _create_var(
        var_name: str, var_attrs: dict, size: Dict[str, int]
    ) -> xarray.Variable:

        du = DatasetUtil()

        # Check variable definition
        TemplateUtil._check_variable_definition(var_name, var_attrs)

        # Unpack variable attributes
        dtype = var_attrs["dtype"]
        dim_names = var_attrs["dim"]
        attributes = var_attrs["attributes"] if "attributes" in var_attrs else None

        err_corr = None
        if attributes is not None:
            if "err_corr" in attributes:
                err_corr = attributes.pop("err_corr")

        # Determine variable shape from dims
        try:
            dim_sizes = TemplateUtil._return_variable_shape(dim_names, size)
        except KeyError:
            raise KeyError(
                "Dim Name Error - Variable "
                + var_name
                + " defined with dim not in dim_sizes_dict"
            )

        # Create variable and add to dataset
        if dtype == str:
            if dtype == "flag":
                flag_meanings = attributes.pop("flag_meanings")
                variable = du.create_flags_variable(
                    dim_sizes,
                    meanings=flag_meanings,
                    dim_names=dim_names,
                    attributes=attributes,
                )

            else:
                raise ValueError("unknown dtype - " + dtype)

        else:
            if err_corr is None:
                variable = du.create_variable(
                    dim_sizes, dim_names=dim_names, dtype=dtype, attributes=attributes
                )

            else:
                variable = du.create_unc_variable(
                    dim_sizes,
                    dim_names=dim_names,
                    dtype=dtype,
                    attributes=attributes,
                    err_corr=err_corr,
                )

            if "encoding" in var_attrs:
                du.add_encoding(variable, **var_attrs["encoding"])

        return variable

    @staticmethod
    def _check_variable_definition(variable_name: str, variable_attrs: Dict):
        """
        Checks validity of variable definition, raising errors as appropriate

        :param variable_name: variable name
        :param variable_attrs: variable defining dictionary
        """

        # Variable name must be type str
        if type(variable_name) != str:
            raise TypeError(
                "Invalid variable name: " + str(variable_name) + " (must be string)"
            )

        # todo - add more tests to check validity of variable definition

    @staticmethod
    def _return_variable_shape(dim_names: List[str], size: Dict[str, int]) -> List[int]:
        """
        Returns dimension sizes of specified dimensions

        :param dim_names: dimension names
        :param size: dictionary of dataset dimensions, entry per dataset dimension with value of size as int

        :returns: dimension sizes
        """

        return [size[dim_name] for dim_name in dim_names]

    @staticmethod
    def add_metadata(ds: xarray.Dataset, metadata: Dict) -> xarray.Dataset:
        """
        Adds metadata to dataset

        :param ds: dataset
        :param metadata: dictionary of dataset metadata

        :returns: dataset with updated metadata
        """

        ds.attrs.update(metadata)

        return ds


if __name__ == "__main__":
    pass
