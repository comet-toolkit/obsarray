"""
Tests for DatasetUtil class
"""

import unittest
import numpy as np
from xarray import DataArray, Variable
from obsarray.templater.dataset_util import DatasetUtil


__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"


class TestDatasetUtil(unittest.TestCase):
    def test_create_default_array_1D_int(self):

        default_array = DatasetUtil.create_default_array([5], np.int8)

        self.assertIsNotNone(default_array)
        self.assertEqual(DataArray, type(default_array))
        self.assertEqual((5,), default_array.shape)
        self.assertEqual(np.int8, default_array.dtype)
        self.assertEqual(-127, default_array[2])

    def test_create_default_array_1D_int_dims(self):

        default_array = DatasetUtil.create_default_array(
            [5], np.int8, dim_names=["dim1"]
        )

        self.assertIsNotNone(default_array)
        self.assertEqual(DataArray, type(default_array))
        self.assertEqual((5,), default_array.shape)
        self.assertEqual(np.int8, default_array.dtype)
        self.assertEqual(-127, default_array[2])
        self.assertEqual(("dim1",), default_array.dims)

    def test_create_default_array_1D_int_fill_value(self):

        default_array = DatasetUtil.create_default_array([5], np.int8, fill_value=1)

        self.assertIsNotNone(default_array)
        self.assertEqual(DataArray, type(default_array))
        self.assertEqual((5,), default_array.shape)
        self.assertEqual(np.int8, default_array.dtype)
        self.assertEqual(1, default_array[2])

    def test_create_default_array_3D_int(self):
        default_array = DatasetUtil.create_default_array([7, 8, 3], np.int8)

        self.assertIsNotNone(default_array)
        self.assertEqual(DataArray, type(default_array))
        self.assertEqual((7, 8, 3), default_array.shape)
        self.assertEqual(np.int8, default_array.dtype)
        self.assertEqual(-127, default_array[2, 4, 2])

    def test_create_default_array_3D_int_dims(self):
        default_array = DatasetUtil.create_default_array(
            [7, 8, 3], np.int8, dim_names=["dim1", "dim2", "dim3"]
        )

        self.assertIsNotNone(default_array)
        self.assertEqual(DataArray, type(default_array))
        self.assertEqual((7, 8, 3), default_array.shape)
        self.assertEqual(np.int8, default_array.dtype)
        self.assertEqual(-127, default_array[2, 4, 2])
        self.assertEqual(
            (
                "dim1",
                "dim2",
                "dim3",
            ),
            default_array.dims,
        )

    def test_create_default_array_3D_int_fill_value(self):
        default_array = DatasetUtil.create_default_array(
            [7, 8, 3], np.int8, fill_value=1
        )

        self.assertIsNotNone(default_array)
        self.assertEqual(DataArray, type(default_array))
        self.assertEqual((7, 8, 3), default_array.shape)
        self.assertEqual(np.int8, default_array.dtype)
        self.assertEqual(1, default_array[2, 4, 2])

    def test_create_variable_1D_int(self):
        vector_variable = DatasetUtil.create_variable([5], np.int8)

        self.assertIsNotNone(vector_variable)
        self.assertEqual(Variable, type(vector_variable))
        self.assertEqual((5,), vector_variable.shape)
        self.assertEqual(np.int8, vector_variable.dtype)
        self.assertEqual(-127, vector_variable[2])

    def test_create_variable_1D_int_dims(self):
        vector_variable = DatasetUtil.create_variable([5], np.int8, dim_names=["dim1"])

        self.assertIsNotNone(vector_variable)
        self.assertEqual(Variable, type(vector_variable))
        self.assertEqual((5,), vector_variable.shape)
        self.assertEqual(np.int8, vector_variable.dtype)
        self.assertEqual(-127, vector_variable[2])
        self.assertEqual(("dim1",), vector_variable.dims)

    def test_create_variable_1D_int_fill_value(self):
        vector_variable = DatasetUtil.create_variable([5], np.int8, fill_value=1)

        self.assertIsNotNone(vector_variable)
        self.assertEqual(Variable, type(vector_variable))
        self.assertEqual((5,), vector_variable.shape)
        self.assertEqual(np.int8, vector_variable.dtype)
        self.assertEqual(1, vector_variable[2])

    def test_create_variable_1D_int_attributes(self):
        vector_variable = DatasetUtil.create_variable(
            [5], np.int8, attributes={"standard_name": "std"}
        )

        self.assertIsNotNone(vector_variable)
        self.assertEqual(Variable, type(vector_variable))
        self.assertEqual((5,), vector_variable.shape)
        self.assertEqual(np.int8, vector_variable.dtype)
        self.assertEqual(-127, vector_variable[2])
        self.assertEqual("std", vector_variable.attrs["standard_name"])

    def test_create_variable_3D_int(self):
        array_variable = DatasetUtil.create_variable([7, 8, 3], np.int8)

        self.assertIsNotNone(array_variable)
        self.assertEqual(Variable, type(array_variable))
        self.assertEqual((7, 8, 3), array_variable.shape)
        self.assertEqual(np.int8, array_variable.dtype)
        self.assertEqual(-127, array_variable[2, 4, 2])

    def test_create_variable_3D_int_dims(self):
        array_variable = DatasetUtil.create_variable(
            [7, 8, 3], np.int8, dim_names=["dim1", "dim2", "dim3"]
        )

        self.assertIsNotNone(array_variable)
        self.assertEqual(Variable, type(array_variable))
        self.assertEqual((7, 8, 3), array_variable.shape)
        self.assertEqual(np.int8, array_variable.dtype)
        self.assertEqual(-127, array_variable[2, 4, 2])
        self.assertEqual(
            (
                "dim1",
                "dim2",
                "dim3",
            ),
            array_variable.dims,
        )

    def test_create_variable_3D_int_fill_value(self):
        array_variable = DatasetUtil.create_variable([7, 8, 3], np.int8, fill_value=1)

        self.assertIsNotNone(array_variable)
        self.assertEqual(Variable, type(array_variable))
        self.assertEqual((7, 8, 3), array_variable.shape)
        self.assertEqual(np.int8, array_variable.dtype)
        self.assertEqual(1, array_variable[2, 4, 2])

    def test_create_variable_3D_int_attributes(self):
        array_variable = DatasetUtil.create_variable(
            [7, 8, 3], np.int8, attributes={"standard_name": "std"}
        )

        self.assertIsNotNone(array_variable)
        self.assertEqual(Variable, type(array_variable))
        self.assertEqual((7, 8, 3), array_variable.shape)
        self.assertEqual(np.int8, array_variable.dtype)
        self.assertEqual(-127, array_variable[2, 4, 2])
        self.assertEqual("std", array_variable.attrs["standard_name"])

    def test_create_unc_variable(self):
        err_corr = [
            {
                "dim": "x",
                "form": "rectangle_absolute",
                "params": [1, 2],
                "units": ["m", "m"],
            },
            {"dim": ["y", "z"], "form": "systematic", "params": [], "units": []},
        ]

        unc_variable = DatasetUtil.create_unc_variable(
            [7, 8, 3, 5],
            np.int8,
            ["x", "y", "z", "a"],
            pdf_shape="gaussian",
            err_corr=err_corr,
        )

        expected_attrs = {
            "err_corr_1_dim": "x",
            "err_corr_1_form": "rectangle_absolute",
            "err_corr_1_units": ["m", "m"],
            "err_corr_1_params": [1, 2],
            "err_corr_2_dim": ["y", "z"],
            "err_corr_2_form": "systematic",
            "err_corr_2_units": [],
            "err_corr_2_params": [],
            "err_corr_3_dim": "a",
            "err_corr_3_form": "random",
            "err_corr_3_units": [],
            "err_corr_3_params": [],
            "pdf_shape": "gaussian",
        }

        self.assertTrue(expected_attrs.items() <= unc_variable.attrs.items())

    def test_create_unc_variable_incorrect_n_fiduceo_params(self):
        err_corr = [
            {
                "dim": "x",
                "form": "rectangle_absolute",
                "params": [1, 2, 3],
                "units": ["m", "m", "m"],
            }
        ]

        self.assertRaises(
            ValueError,
            DatasetUtil.create_unc_variable,
            [7],
            np.int8,
            ["x"],
            pdf_shape="gaussian",
            err_corr=err_corr,
        )

        pass

    def test_create_flags_variable_1D(self):

        meanings = [
            "flag1",
            "flag2",
            "flag3",
            "flag4",
            "flag5",
            "flag6",
            "flag7",
            "flag8",
        ]
        meanings_txt = "flag1 flag2 flag3 flag4 flag5 flag6 flag7 flag8"
        masks = "1, 2, 4, 8, 16, 32, 64, 128"
        flags_vector_variable = DatasetUtil.create_flags_variable(
            [5], meanings, dim_names=["dim1"], attributes={"standard_name": "std"}
        )

        self.assertIsNotNone(flags_vector_variable)
        self.assertEqual(Variable, type(flags_vector_variable))
        self.assertEqual((5,), flags_vector_variable.shape)
        self.assertEqual(np.uint8, flags_vector_variable.dtype)
        self.assertEqual(flags_vector_variable.attrs["flag_masks"], masks)
        self.assertEqual(flags_vector_variable.attrs["flag_meanings"], meanings_txt)
        self.assertEqual(0, flags_vector_variable[2])
        self.assertEqual("std", flags_vector_variable.attrs["standard_name"])
        self.assertEqual(("dim1",), flags_vector_variable.dims)

    def test_create_flags_variable_3D(self):

        meanings = [
            "flag1",
            "flag2",
            "flag3",
            "flag4",
            "flag5",
            "flag6",
            "flag7",
            "flag8",
        ]
        meanings_txt = "flag1 flag2 flag3 flag4 flag5 flag6 flag7 flag8"
        masks = "1, 2, 4, 8, 16, 32, 64, 128"
        flags_array_variable = DatasetUtil.create_flags_variable(
            [7, 8, 3],
            meanings,
            dim_names=["dim1", "dim2", "dim3"],
            attributes={"standard_name": "std"},
        )

        self.assertIsNotNone(flags_array_variable)
        self.assertEqual(Variable, type(flags_array_variable))
        self.assertEqual((7, 8, 3), flags_array_variable.shape)
        self.assertEqual(np.uint8, flags_array_variable.dtype)
        self.assertEqual(flags_array_variable.attrs["flag_masks"], masks)
        self.assertEqual(flags_array_variable.attrs["flag_meanings"], meanings_txt)
        self.assertEqual(0, flags_array_variable[2, 4, 2])
        self.assertEqual("std", flags_array_variable.attrs["standard_name"])
        self.assertEqual(("dim1", "dim2", "dim3"), flags_array_variable.dims)

    def test_return_flags_dtype_5(self):
        data_type = DatasetUtil.return_flags_dtype(5)
        self.assertEqual(data_type, np.uint8)

    def test_return_flags_dtype_8(self):
        data_type = DatasetUtil.return_flags_dtype(8)
        self.assertEqual(data_type, np.uint8)

    def test_return_flags_dtype_15(self):
        data_type = DatasetUtil.return_flags_dtype(15)
        self.assertEqual(data_type, np.uint16)

    def test_return_flags_dtype_16(self):
        data_type = DatasetUtil.return_flags_dtype(16)
        self.assertEqual(data_type, np.uint16)

    def test_return_flags_dtype_17(self):
        data_type = DatasetUtil.return_flags_dtype(17)
        self.assertEqual(data_type, np.uint32)

    def test_return_flags_dtype_32(self):
        data_type = DatasetUtil.return_flags_dtype(32)
        self.assertEqual(data_type, np.uint32)

    def test_add_encoding(self):
        vector_variable = DatasetUtil.create_variable([5], np.int8)
        DatasetUtil.add_encoding(
            vector_variable,
            np.int32,
            scale_factor=10,
            offset=23,
            fill_value=11,
            chunksizes=12,
        )

        self.assertIsNotNone(vector_variable)
        self.assertEqual(Variable, type(vector_variable))
        self.assertEqual((5,), vector_variable.shape)
        self.assertEqual(np.int8, vector_variable.dtype)
        self.assertEqual(-127, vector_variable[2])

        self.assertEqual(np.int32, vector_variable.encoding["dtype"])
        self.assertEqual(10, vector_variable.encoding["scale_factor"])
        self.assertEqual(23, vector_variable.encoding["add_offset"])
        self.assertEqual(11, vector_variable.encoding["_FillValue"])
        self.assertEqual(12, vector_variable.encoding["chunksizes"])

    def test_get_default_fill_value(self):

        self.assertEqual(-127, DatasetUtil.get_default_fill_value(np.int8))
        self.assertEqual(-32767, DatasetUtil.get_default_fill_value(np.int16))
        self.assertEqual(np.uint16(-1), DatasetUtil.get_default_fill_value(np.uint16))
        self.assertEqual(-2147483647, DatasetUtil.get_default_fill_value(np.int32))
        self.assertEqual(
            -9223372036854775806, DatasetUtil.get_default_fill_value(np.int64)
        )
        self.assertEqual(
            np.float32(9.96921e36), DatasetUtil.get_default_fill_value(np.float32)
        )
        self.assertEqual(
            9.969209968386869e36, DatasetUtil.get_default_fill_value(np.float64)
        )


if __name__ == "__main__":
    unittest.main()
