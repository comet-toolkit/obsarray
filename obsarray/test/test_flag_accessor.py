"""test_flag_accessor - tests for obsarray.flag_accessor"""

import xarray as xr
import pandas as pd
import numpy as np
import unittest
from unittest.mock import patch
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
            pressure=(["x", "y", "time"], temperature, {"units": "K"}),
            general_flags=(
                ["x", "y", "time"],
                np.zeros(temperature.shape, dtype=np.int8),
                {"flag_meanings": []},
            ),
            temperature_flags=(
                ["x", "y", "time"],
                np.zeros(temperature.shape, dtype=np.int8),
                {
                    "flag_meanings": ["bad_data", "dubious data"],
                    "applicable_vars": ["temperature"],
                },
            ),
        ),
        coords=dict(
            lon=(["x", "y"], lon),
            lat=(["x", "y"], lat),
            time=time,
            reference_time=reference_time,
        ),
        attrs=dict(description="Weather related data."),
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
        self.assertCountEqual(list(data_vars.variables), ["temperature", "pressure"])

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

    def test___len__(self):
        self.assertEqual(len(self.ds.flag["temperature_flags"]), 2)

    def test___iter__(self):

        var_names = []
        for flag in self.ds.flag["temperature_flags"]:
            self.assertIsInstance(flag, obsarray.flag_accessor.Flag)
            var_names.append(flag._flag_meaning)

        self.assertCountEqual(var_names, ["bad_data", "dubious data"])

    def test_keys(self):
        self.assertCountEqual(
            self.ds.flag["temperature_flags"].keys(),
            ["bad_data", "dubious data"],
        )


class TestFlag(unittest.TestCase):
    def setUp(self):
        self.ds = create_ds()

    @patch("obsarray.flag_accessor.Flag.expand_sli", return_value="slice")
    def test___getitem__(self, m):
        self.assertEqual(
            self.ds.flag["temperature_flags"]["bad_data"]["in_slice"]._sli, "slice"
        )

        m.assert_called_with("in_slice")

    def test_expand_slice_full(self):
        sli = self.ds.flag["temperature_flags"]["bad_data"].expand_sli((1, 1, 1))
        self.assertEqual((1, 1, 1), sli)

    def test_expand_slice_None(self):
        sli = self.ds.flag["temperature_flags"]["bad_data"].expand_sli()
        self.assertEqual((slice(None), slice(None), slice(None)), sli)

    def test_expand_slice_first(self):
        sli = self.ds.flag["temperature_flags"]["bad_data"].expand_sli((0,))
        self.assertEqual((0, slice(None), slice(None)), sli)


if __name__ == "__main__":
    unittest.main()
