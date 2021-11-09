"""
Tests for TemplateUtil class
"""

import unittest
from unittest.mock import patch, call
import numpy as np
import xarray
from obsarray.templater.template_util import TemplateUtil, create_ds


__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"


class TestTemplateUtil(unittest.TestCase):
    @patch("obsarray.templater.template_util.DatasetUtil")
    def test_add_variables_1var(self, mock_du):
        dataset = xarray.Dataset()

        size = {"dim1": 25, "dim2": 30, "dim3": 10, "dim4": 15}

        template = {
            "array_variable": {
                "dim": ["dim1", "dim2"],
                "dtype": np.float32,
                "attributes": {
                    "standard_name": "array_variable_std_name",
                    "long_name": "array_variable_long_name",
                    "units": "units",
                    "preferred_symbol": "av",
                },
                "encoding": {"dtype": np.uint16, "scale_factor": 1.0, "offset": 0.0},
            }
        }

        # run method
        dataset = TemplateUtil.add_variables(dataset, template, size)

        # test results
        mock_du.return_value.create_variable.assert_called_once_with(
            [25, 30],
            dim_names=["dim1", "dim2"],
            dtype=np.float32,
            attributes={
                "standard_name": "array_variable_std_name",
                "long_name": "array_variable_long_name",
                "units": "units",
                "preferred_symbol": "av",
            },
        )
        mock_du.return_value.add_encoding.assert_called_once_with(
            mock_du.return_value.create_variable.return_value,
            dtype=np.uint16,
            scale_factor=1.0,
            offset=0.0,
        )

    @patch("obsarray.templater.template_util.DatasetUtil")
    def test_add_variables_1uncvar(self, mock_du):
        dataset = xarray.Dataset()

        size = {"dim1": 25, "dim2": 30, "dim3": 10, "dim4": 15}

        template = {
            "array_variable": {
                "dim": ["dim1", "dim2"],
                "dtype": np.float32,
                "attributes": {
                    "standard_name": "array_variable_std_name",
                    "long_name": "array_variable_long_name",
                    "units": "units",
                    "preferred_symbol": "av",
                    "err_corr": "err_corr",
                },
                "encoding": {"dtype": np.uint16, "scale_factor": 1.0, "offset": 0.0},
            }
        }

        # run method
        dataset = TemplateUtil.add_variables(dataset, template, size)

        # test results
        mock_du.return_value.create_unc_variable.assert_called_once_with(
            [25, 30],
            dim_names=["dim1", "dim2"],
            dtype=np.float32,
            attributes={
                "standard_name": "array_variable_std_name",
                "long_name": "array_variable_long_name",
                "units": "units",
                "preferred_symbol": "av",
            },
            err_corr="err_corr",
        )
        mock_du.return_value.add_encoding.assert_called_once_with(
            mock_du.return_value.create_unc_variable.return_value,
            dtype=np.uint16,
            scale_factor=1.0,
            offset=0.0,
        )

    @patch("obsarray.templater.template_util.DatasetUtil")
    def test_add_variables_1flag(self, mock_du):
        dataset = xarray.Dataset()

        dim_sizes = {"dim1": 25, "dim2": 30, "dim3": 10, "dim4": 15}

        test_variables = {
            "array_variable": {
                "dim": ["dim1", "dim2"],
                "dtype": "flag",
                "attributes": {
                    "standard_name": "array_variable_std_name",
                    "long_name": "array_variable_long_name",
                    "preferred_symbol": "av",
                    "flag_meanings": ["flag1"],
                },
            }
        }

        # run method
        dataset = TemplateUtil.add_variables(dataset, test_variables, dim_sizes)

        # test results
        mock_du.return_value.create_flags_variable.assert_called_once_with(
            [25, 30],
            dim_names=["dim1", "dim2"],
            meanings=["flag1"],
            attributes={
                "standard_name": "array_variable_std_name",
                "long_name": "array_variable_long_name",
                "preferred_symbol": "av",
            },
        )

    @patch("obsarray.templater.template_util.DatasetUtil")
    def test_add_variables_2var(self, mock_du):
        dataset = xarray.Dataset()

        size = {"dim1": 25, "dim2": 30, "dim3": 10, "dim4": 15}

        template = {
            "array_variable1": {
                "dim": ["dim1", "dim2"],
                "dtype": np.float32,
                "attributes": {
                    "standard_name": "array_variable_std_name1",
                    "long_name": "array_variable_long_name1",
                    "units": "units",
                    "preferred_symbol": "av",
                },
                "encoding": {"dtype": np.uint16, "scale_factor": 1.0, "offset": 0.0},
            },
            "array_variable2": {
                "dim": ["dim3", "dim4"],
                "dtype": np.float32,
                "attributes": {
                    "standard_name": "array_variable_std_name2",
                    "long_name": "array_variable_long_name2",
                    "units": "units",
                    "preferred_symbol": "av",
                },
                "encoding": {"dtype": np.uint16, "scale_factor": 1.0, "offset": 0.0},
            },
        }

        # run method
        dataset = TemplateUtil.add_variables(dataset, template, size)

        # test results
        # define expected calls to DatasetUtil methods
        calls = [
            call(
                [25, 30],
                dim_names=["dim1", "dim2"],
                dtype=np.float32,
                attributes={
                    "standard_name": "array_variable_std_name1",
                    "long_name": "array_variable_long_name1",
                    "units": "units",
                    "preferred_symbol": "av",
                },
            ),
            call(
                [10, 15],
                dim_names=["dim3", "dim4"],
                dtype=np.float32,
                attributes={
                    "standard_name": "array_variable_std_name2",
                    "long_name": "array_variable_long_name2",
                    "units": "units",
                    "preferred_symbol": "av",
                },
            ),
        ]

        mock_du.return_value.create_variable.assert_has_calls(calls, any_order=True)

    def test_add_variables_1var_ds(self):
        dataset = xarray.Dataset()

        size = {"dim1": 25, "dim2": 30, "dim3": 10, "dim4": 15}

        template = {
            "array_variable": {
                "dim": ["dim1", "dim2"],
                "dtype": np.float32,
                "attributes": {
                    "standard_name": "array_variable_std_name",
                    "long_name": "array_variable_long_name",
                    "units": "units",
                    "preferred_symbol": "av",
                },
                "encoding": {"dtype": np.uint16, "scale_factor": 1.0, "offset": 0.0},
            }
        }

        # run method
        dataset = TemplateUtil.add_variables(dataset, template, size)

        # assert dataset with variables
        self.assertEqual(type(dataset), xarray.Dataset)
        self.assertEqual(type(dataset["array_variable"]), xarray.DataArray)

    def test__return_variable_shape(self):
        size = {"dim1": 25, "dim2": 30, "dim3": 10, "dim4": 15}
        dim_names = ["dim2", "dim1"]

        dim_sizes = TemplateUtil._return_variable_shape(dim_names=dim_names, size=size)

        self.assertCountEqual([30, 25], dim_sizes)

    def test__check_variable_definition_bad_name(self):
        test_variable_name = 23
        test_variable_attrs = {
            "dim": ["dim1", "dim2"],
            "dtype": np.float32,
            "attributes": {
                "standard_name": "array_variable_std_name",
                "long_name": "array_variable_long_name",
                "units": "units",
                "preferred_symbol": "av",
            },
            "encoding": {"dtype": np.uint16, "scale_factor": 1.0, "offset": 0.0},
        }

        self.assertRaises(
            TypeError,
            TemplateUtil._check_variable_definition,
            test_variable_name,
            test_variable_attrs,
        )

    def test_add_metadata(self):
        dataset = xarray.Dataset()

        test_metadata = {"metadata1": "value"}

        TemplateUtil.add_metadata(dataset, test_metadata)

        self.assertEqual("value", dataset.attrs["metadata1"])

    def test_create_template_dataset(self):
        dim_sizes = {"dim1": 25, "dim2": 30, "dim3": 10, "dim4": 15}

        test_variables = {
            "array_variable": {
                "dim": ["dim1", "dim2"],
                "dtype": np.float32,
                "attributes": {
                    "standard_name": "array_variable_std_name",
                    "long_name": "array_variable_long_name",
                    "units": "units",
                    "preferred_symbol": "av",
                },
                "encoding": {"dtype": np.uint16, "scale_factor": 1.0, "offset": 0.0},
            }
        }

        test_metadata = {"metadata1": "value"}

        ds = create_ds(test_variables, dim_sizes, test_metadata)

        self.assertEqual(type(ds), xarray.Dataset)
        self.assertEqual(type(ds["array_variable"]), xarray.DataArray)
        self.assertEqual("value", ds.attrs["metadata1"])


if __name__ == "__main__":
    unittest.main()
