"""test_flag_accessor - tests for obsarray.flag_accessor"""

from copy import deepcopy
import xarray as xr
import pandas as pd
import numpy as np
import unittest
from unittest.mock import patch
from obsarray.templater.template_util import create_var
import obsarray


__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"
__all__ = []


def create_ds():
    np.random.seed(0)
    temperature = 15 + 8 * np.random.randn(2, 2, 3)
    lon = [[-99.83, -99.32], [-99.79, -99.23]]
    lat = [[42.25, 42.21], [42.63, 42.59]]
    time = pd.date_range("2014-09-06", periods=3)
    reference_time = pd.Timestamp("2014-09-05")

    ds = xr.Dataset(
        data_vars=dict(
            temperature=(["x", "y", "time"], temperature, {"units": "K"}),
            precipitation=(["x", "y", "time"], temperature, {"units": "K"}),
        ),
        coords=dict(
            lon=(["x", "y"], lon),
            lat=(["x", "y"], lat),
            time=time,
            reference_time=reference_time,
        ),
        attrs=dict(description="Weather related data."),
    )

    ds["temperature_flags"] = create_var(
        "temperature_flags",
        {
            "dtype": "flag",
            "dim": ["x", "y", "time"],
            "attributes": {
                "flag_meanings": ["bad_data", "dubious_data"],
                "applicable_vars": ["temperature"],
            },
        },
        {"x": 2, "y": 2, "time": 3},
    )

    ds["general_flags"] = create_var(
        "general_flags",
        {
            "dtype": "flag",
            "dim": ["x", "y", "time"],
            "attributes": {"flag_meanings": []},
        },
        {"x": 2, "y": 2, "time": 3},
    )

    ds["time_flags"] = create_var(
        "temperature_flags",
        {
            "dtype": "flag",
            "dim": ["time"],
            "attributes": {"flag_meanings": ["dubious", "invalid"]},
        },
        {"time": 3},
    )

    return ds


class TestFlagAccessor(unittest.TestCase):
    def setUp(self):
        self.ds = create_ds()

    def test___getitem__(self):
        self.assertIsInstance(
            self.ds.flag["general_flag"], obsarray.flag_accessor.FlagVariable
        )
        self.assertEqual(self.ds.flag["general_flag"]._flag_var_name, "general_flag")

    def test___len__(self):
        self.assertEqual(len(self.ds.flag), 2)

    def test___iter__(self):

        var_names = []
        for flag in self.ds.flag:
            self.assertIsInstance(flag, obsarray.flag_accessor.FlagVariable)
            var_names.append(flag._flag_var_name)

        self.assertCountEqual(var_names, ["temperature_flags", "general_flags"])

    def test_keys(self):
        self.assertCountEqual(
            self.ds.flag.keys(), ["temperature_flags", "general_flags"]
        )

    def test__is_data_var_true(self):
        self.assertTrue(self.ds.flag._is_data_var("temperature"))

    def test__is_data_var_false(self):
        self.assertFalse(self.ds.flag._is_data_var("temperature_flags"))

    def test_data_vars(self):
        data_vars = self.ds.flag.data_vars
        self.assertEqual(type(data_vars), xr.core.dataset.DataVariables)
        self.assertCountEqual(
            list(data_vars.variables), ["temperature", "precipitation"]
        )

    def test_flag_vars(self):
        flag_vars = self.ds.flag.flag_vars
        self.assertEqual(type(flag_vars), xr.core.dataset.DataVariables)
        self.assertCountEqual(
            list(flag_vars.variables),
            ["temperature_flags", "general_flags"],
        )

    def test__is_flag_var_true(self):
        self.assertTrue(self.ds.flag._is_flag_var("temperature_flags"))

    def test__is_flag_var_false(self):
        self.assertFalse(self.ds.flag._is_flag_var("temperature"))

    def test___setitem___DataArray(self):
        new_flag = xr.DataArray(
            np.zeros(self.ds.precipitation.shape),
            dims=["x", "y", "time"],
            attrs={"flag_meanings": "test"},
        )

        self.ds.flag["new_flag"] = new_flag

        self.assertEqual(self.ds.new_flag.attrs["flag_meanings"], "test")
        self.assertEqual(self.ds.new_flag.attrs["flag_masks"], "1")

        np.testing.assert_array_equal(new_flag.values, self.ds.new_flag.values)

    def test___setitem___DataArray_nomeanings(self):
        new_flag = xr.DataArray(
            np.zeros(self.ds.precipitation.shape), dims=["x", "y", "time"], attrs={}
        )

        self.ds.flag["new_flag"] = new_flag

        self.assertEqual(self.ds.new_flag.attrs["flag_meanings"], "")
        self.assertEqual(self.ds.new_flag.attrs["flag_masks"], "")

        np.testing.assert_array_equal(new_flag.values, self.ds.new_flag.values)

    @patch("obsarray.flag_accessor.create_var")
    def test___setitem___tuple(self, mock):

        flag_def = (
            ["x", "y", "time"],
            {"flag_meanings": ["flag1", "flag2", "flag3"]},
        )
        self.ds.flag["new_flag"] = flag_def

        mock.assert_called_once_with(
            "new_flag",
            {
                "dtype": "flag",
                "dim": ["x", "y", "time"],
                "attributes": {"flag_meanings": ["flag1", "flag2", "flag3"]},
            },
            {"x": 2, "y": 2, "time": 3},
        )

        self.assertTrue(("new_flag" in self.ds))


class TestFlagVariable(unittest.TestCase):
    def setUp(self):
        self.ds = create_ds()

    def test___getitem__(self):
        self.assertIsInstance(
            self.ds.flag["temperature_flags"]["bad_data"],
            obsarray.flag_accessor.Flag,
        )
        self.assertEqual(
            self.ds.flag["temperature_flags"]["bad_data"]._flag_var_name,
            "temperature_flags",
        )
        self.assertEqual(
            self.ds.flag["temperature_flags"]["bad_data"]._flag_meaning,
            "bad_data",
        )

    @patch("obsarray.flag_accessor.Flag.__setitem__")
    @patch("obsarray.flag_accessor.DatasetUtil.add_flag_meaning_to_attrs")
    def test___setitem___existing_mask(self, mdu, mf):
        self.ds.flag["temperature_flags"]["bad_data"] = True
        mdu.assert_not_called()
        mf.assert_called_with(slice(None, None, None), True)

    @patch("obsarray.flag_accessor.Flag.__setitem__")
    @patch("obsarray.flag_accessor.DatasetUtil.add_flag_meaning_to_attrs")
    def test___setitem___new_mask_1d(self, mdu, mf):
        original_attrs = deepcopy(self.ds["time_flags"].attrs)

        self.ds.flag["time_flags"]["test_flag"] = True

        mdu.assert_called_once_with(
            original_attrs, "test_flag", self.ds["time_flags"].dtype
        )

        self.assertDictEqual(self.ds["time_flags"].attrs, {})

        mf.assert_called_with(slice(None, None, None), True)

    @patch("obsarray.flag_accessor.Flag.__setitem__")
    @patch("obsarray.flag_accessor.DatasetUtil.add_flag_meaning_to_attrs")
    def test___setitem___new_mask(self, mdu, mf):
        original_attrs = deepcopy(self.ds["temperature_flags"].attrs)

        self.ds.flag["temperature_flags"]["test_flag"] = True

        mdu.assert_called_once_with(
            original_attrs, "test_flag", self.ds["temperature_flags"].dtype
        )

        self.assertDictEqual(self.ds["temperature_flags"].attrs, {})

        mf.assert_called_with(slice(None, None, None), True)

    @patch("obsarray.flag_accessor.Flag.__setitem__")
    @patch("obsarray.flag_accessor.DatasetUtil.add_flag_meaning_to_attrs")
    def test___setitem___new_mask(self, mdu, mf):

        original_attrs = deepcopy(self.ds["temperature_flags"].attrs)

        self.ds.flag["temperature_flags"]["test_flag"] = True

        mdu.assert_called_once_with(
            original_attrs, "test_flag", self.ds["temperature_flags"].dtype
        )

        self.assertDictEqual(self.ds["temperature_flags"].attrs, {})

        mf.assert_called_with(slice(None, None, None), True)

    @patch("obsarray.flag_accessor.Flag.__setitem__")
    @patch("obsarray.flag_accessor.DatasetUtil.rm_flag_meaning_from_attrs")
    def test___delitem__(self, mdu, mf):
        original_attrs = deepcopy(self.ds["temperature_flags"].attrs)

        del self.ds.flag["temperature_flags"]["dubious_data"]

        mdu.assert_called_once_with(original_attrs, "dubious_data")

        self.assertDictEqual(self.ds["temperature_flags"].attrs, {})

        mf.assert_called_with(slice(None, None, None), False)

    def test___len__(self):
        self.assertEqual(len(self.ds.flag["temperature_flags"]), 2)

    def test___iter__(self):

        var_names = []
        for flag in self.ds.flag["temperature_flags"]:
            self.assertIsInstance(flag, obsarray.flag_accessor.Flag)
            var_names.append(flag._flag_meaning)

        self.assertCountEqual(var_names, ["bad_data", "dubious_data"])

    def test_keys(self):
        self.assertCountEqual(
            self.ds.flag["temperature_flags"].keys(),
            ["bad_data", "dubious_data"],
        )


class TestFlag(unittest.TestCase):
    def setUp(self):
        self.ds = create_ds()

    @patch("obsarray.flag_accessor.Flag._expand_sli", return_value="slice")
    def test___getitem__(self, m):
        self.assertEqual(
            self.ds.flag["temperature_flags"]["bad_data"]["in_slice"]._sli, "slice"
        )

        m.assert_called_with("in_slice")

    def test_expand_slice_1d_full(self):
        sli = self.ds.flag["time_flags"]["dubious"]._expand_sli((1))
        self.assertEqual((1,), sli)

    def test_expand_slice_1d_None(self):
        sli = self.ds.flag["time_flags"]["dubious"]._expand_sli()
        self.assertEqual((slice(None),), sli)

    def test_expand_slice_full(self):
        sli = self.ds.flag["temperature_flags"]["bad_data"]._expand_sli((1, 1, 1))
        self.assertEqual((1, 1, 1), sli)

    def test_expand_slice_None(self):
        sli = self.ds.flag["temperature_flags"]["bad_data"]._expand_sli()
        self.assertEqual((slice(None), slice(None), slice(None)), sli)

    def test_expand_slice_first(self):
        sli = self.ds.flag["temperature_flags"]["bad_data"]._expand_sli((0,))
        self.assertEqual((0, slice(None), slice(None)), sli)

    def test__setitem___1element_False2True(self):
        self.ds["temperature_flags"].values[:] = 0

        self.ds.flag["temperature_flags"]["bad_data"][0, 0, 0] = True

        exp_flags = np.array([[[1, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]])

        np.testing.assert_array_equal(self.ds["temperature_flags"].values, exp_flags)

    def test__setitem___False2True(self):
        self.ds.flag["temperature_flags"]["bad_data"][:, 0, :] = True

        exp_flags = np.array([[[1, 1, 1], [0, 0, 0]], [[1, 1, 1], [0, 0, 0]]])

        np.testing.assert_array_equal(self.ds["temperature_flags"].values, exp_flags)

    def test__setitem___True2False(self):

        self.ds["temperature_flags"].values[:] = 1

        self.ds.flag["temperature_flags"]["bad_data"][:, 1, :] = False

        exp_flags = np.array([[[1, 1, 1], [0, 0, 0]], [[1, 1, 1], [0, 0, 0]]])

        np.testing.assert_array_equal(self.ds["temperature_flags"].values, exp_flags)

    def test__setitem___array(self):
        self.ds["temperature_flags"].values[:] = np.array(
            [[[1, 1, 1], [0, 0, 0]], [[1, 1, 1], [0, 0, 0]]]
        )

        self.ds.flag["temperature_flags"]["bad_data"][:, :, 0] = np.array(
            [[False, True], [False, True]]
        )

        exp_flags = np.array([[[0, 1, 1], [1, 0, 0]], [[0, 1, 1], [1, 0, 0]]])

        np.testing.assert_array_equal(self.ds["temperature_flags"].values, exp_flags)

    def test__setitem___array_not_bool(self):
        self.assertRaises(
            TypeError,
            self.ds.flag["temperature_flags"]["bad_data"],
            slice(None, None, 0),
            np.array([[0.0, 1.0], [0.0, 1.0]]),
        )

    def test_value(self):
        self.ds["temperature_flags"].values[:] = np.array(
            [[[1, 1, 1], [0, 0, 0]], [[1, 1, 1], [0, 0, 0]]]
        )

        mask = self.ds.flag["temperature_flags"]["bad_data"][:, :, 0].value
        exp_mask = np.array([[True, False], [True, False]])

        np.testing.assert_array_equal(mask, exp_mask)


if __name__ == "__main__":
    unittest.main()
