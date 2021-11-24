"""test_unc_accessor - tests for obsarray.unc_accessor"""

import xarray as xr
import pandas as pd
import numpy as np
import unittest
from unittest.mock import patch
import obsarray


__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"
__all__ = []


class TestUncAccessor(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        temperature = 15 + 8 * np.random.randn(2, 2, 3)
        u_r_temperature = temperature * 0.02
        u_s_temperature = temperature * 0.01
        precipiation = 15 + 8 * np.random.randn(2, 2, 3)
        u_r_precipiation = temperature * 0.02
        lon = [[-99.83, -99.32], [-99.79, -99.23]]
        lat = [[42.25, 42.21], [42.63, 42.59]]
        time = pd.date_range("2014-09-06", periods=3)
        reference_time = pd.Timestamp("2014-09-05")

        self.ds = xr.Dataset(
            data_vars=dict(
                temperature=(["x", "y", "time"], temperature),
                u_r_temperature=(["x", "y", "time"], u_r_temperature),
                u_s_temperature=(["x", "y", "time"], u_s_temperature),
                precipitation=(["x", "y", "time"], precipiation),
                u_r_precipitation=(["x", "y", "time"], u_r_precipiation),
            ),
            coords=dict(
                lon=(["x", "y"], lon),
                lat=(["x", "y"], lat),
                time=time,
                reference_time=reference_time,
            ),
            attrs=dict(description="Weather related data."),
        )

        self.ds.temperature.attrs["unc_comps"] = ["u_r_temperature", "u_s_temperature"]
        self.ds.precipitation.attrs["unc_comps"] = ["u_r_precipitation"]

    def test__var_unc_var_names(self):
        self.assertCountEqual(
            self.ds.unc._var_unc_var_names("temperature"),
            ["u_r_temperature", "u_s_temperature"],
        )

    def test__var_unc_var_names_none(self):
        self.assertCountEqual(self.ds.unc._var_unc_var_names("u_r_temperature"), [])

    def test__is_obs_var_true(self):
        self.assertTrue(self.ds.unc._is_obs_var("temperature"))

    def test__is_obs_var_false(self):
        self.assertFalse(self.ds.unc._is_obs_var("u_r_temperature"))

    def test_obs_vars(self):
        obs_vars = self.ds.unc.obs_vars
        self.assertEqual(type(obs_vars), xr.core.dataset.DataVariables)
        self.assertCountEqual(
            list(obs_vars.variables), ["temperature", "precipitation"]
        )

    def test_unc_vars(self):
        unc_vars = self.ds.unc.unc_vars
        self.assertEqual(type(unc_vars), xr.core.dataset.DataVariables)
        self.assertCountEqual(
            list(unc_vars.variables),
            ["u_r_temperature", "u_s_temperature", "u_r_precipitation"],
        )

    def test__is_unc_var_true(self):
        self.assertTrue(self.ds.unc._is_unc_var("u_r_temperature"))

    def test__is_unc_var_false(self):
        self.assertFalse(self.ds.unc._is_unc_var("temperature"))

    def test__var_unc_vars(self):
        unc_vars = self.ds.unc._var_unc_vars("temperature")
        self.assertEqual(type(unc_vars), xr.core.dataset.DataVariables)
        self.assertCountEqual(
            list(unc_vars.variables), ["u_r_temperature", "u_s_temperature"]
        )

    def test__var_unc_vars_none(self):
        unc_vars = self.ds.unc._var_unc_vars("u_r_temperature")
        self.assertEqual(type(unc_vars), xr.core.dataset.DataVariables)
        self.assertCountEqual(list(unc_vars.variables), [])

    def test__quadsum(self):

        test_ds = xr.Dataset(
            data_vars=dict(
                a=(["x"], np.ones(3) * 3),
                b=(["x"], np.ones(3) * 4),
            )
        )
        c = test_ds.unc._quadsum()

        np.testing.assert_array_equal(c.values, np.ones(3) * 5)

    def test__add_unc_var_DataArray(self):
        u_s_precipitation = xr.DataArray(
            self.ds.precipitation.values, dims=["x", "y", "time"]
        )
        self.ds.unc._add_unc_var(
            "precipitation", "u_s_precipitation", u_s_precipitation
        )
        self.assertTrue(
            "u_s_precipitation" in self.ds.unc._var_unc_var_names("precipitation")
        )

        np.testing.assert_array_equal(
            u_s_precipitation.values, self.ds.u_s_precipitation.values
        )

    @patch("obsarray.unc_accessor.create_var")
    def test__add_unc_var_tuple(self, mock):

        unc_def = (
            ["x", "y", "time"],
            self.ds.precipitation.values,
            {"err_corr": []},
        )
        self.ds.unc._add_unc_var("precipitation", "u_s_precipitation", unc_def)
        self.assertTrue(
            "u_s_precipitation" in self.ds.unc._var_unc_var_names("precipitation")
        )

        mock.assert_called_once_with(
            "u_s_precipitation",
            {
                "dtype": np.float64,
                "dim": ["x", "y", "time"],
                "attributes": {"err_corr": []},
            },
            {"x": 2, "y": 2, "time": 3},
        )

        self.assertTrue(("u_s_precipitation" in self.ds))

    @patch("obsarray.unc_accessor.create_var")
    def test__add_unc_var_tuple_no_err_corr(self, mock):
        unc_def = (["x", "y", "time"], self.ds.precipitation.values, {})
        self.ds.unc._add_unc_var("precipitation", "u_s_precipitation", unc_def)
        self.assertTrue(
            "u_s_precipitation" in self.ds.unc._var_unc_var_names("precipitation")
        )

        mock.assert_called_once_with(
            "u_s_precipitation",
            {
                "dtype": np.float64,
                "dim": ["x", "y", "time"],
                "attributes": {"err_corr": []},
            },
            {"x": 2, "y": 2, "time": 3},
        )

        self.assertTrue(("u_s_precipitation" in self.ds))

    @patch("obsarray.unc_accessor.create_var")
    def test__add_unc_var_tuple_no_attrs(self, mock):
        unc_def = (["x", "y", "time"], self.ds.precipitation.values)
        self.ds.unc._add_unc_var("precipitation", "u_s_precipitation", unc_def)
        self.assertTrue(
            "u_s_precipitation" in self.ds.unc._var_unc_var_names("precipitation")
        )

        mock.assert_called_once_with(
            "u_s_precipitation",
            {
                "dtype": np.float64,
                "dim": ["x", "y", "time"],
                "attributes": {"err_corr": []},
            },
            {"x": 2, "y": 2, "time": 3},
        )

        "u_s_precipitation", self.ds.unc["precipitation"].keys()

        self.assertTrue(("u_s_precipitation" in self.ds))

    # def test_total(self):
    #     u_tot_temperature = self.ds.unc.total("temperature")
    #     u_tot_temperature_test = (self.ds.u_r_temperature ** 2. + self.ds.u_s_temperature ** 2.0) ** 0.5
    #
    #     np.testing.assert_array_equal(u_tot_temperature.values, u_tot_temperature_test.values)


class TestErrCorr(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        temperature = 15 + 8 * np.random.randn(2, 2, 3)
        u_r_temperature = temperature * 0.02
        u_s_temperature = temperature * 0.01
        precipiation = 15 + 8 * np.random.randn(2, 2, 3)
        u_r_precipiation = temperature * 0.02
        lon = [[-99.83, -99.32], [-99.79, -99.23]]
        lat = [[42.25, 42.21], [42.63, 42.59]]
        time = pd.date_range("2014-09-06", periods=3)
        reference_time = pd.Timestamp("2014-09-05")

        self.ds = xr.Dataset(
            data_vars=dict(
                temperature=(["x", "y", "time"], temperature),
            ),
            coords=dict(
                lon=(["x", "y"], lon),
                lat=(["x", "y"], lat),
                time=time,
                reference_time=reference_time,
            ),
            attrs=dict(description="Weather related data."),
        )

        self.ds.unc["temperature"]["u_ran_temperature"] = (
            ["x", "y", "time"],
            temperature * 0.05,
        )

        self.ds.unc["temperature"]["u_sys_temperature"] = (
            ["x", "y", "time"],
            temperature * 0.03,
            {
                "err_corr": [
                    {
                        "dim": "x",
                        "form": "systematic",
                        "params": [],
                    },
                    {
                        "dim": "y",
                        "form": "systematic",
                        "params": [],
                    },
                    {
                        "dim": "time",
                        "form": "systematic",
                        "params": [],
                    },
                ]
            },
        )

        self.ds.unc["temperature"]["u_str_temperature"] = (
            ["x", "y", "time"],
            temperature * 0.03,
            {
                "err_corr": [
                    {
                        "dim": "x",
                        "form": "custom",
                        "params": ["err_corr_str_temperature"],
                    },
                    {
                        "dim": "y",
                        "form": "systematic",
                        "params": [],
                    },
                    {
                        "dim": "time",
                        "form": "systematic",
                        "params": [],
                    },
                ]
            },
        )

    def test_to_dict(self):
        self.ds.unc.__str__()
        pass


if __name__ == "__main__":
    unittest.main()
