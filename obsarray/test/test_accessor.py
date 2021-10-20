"""test_accessor - tests for obsarray.accessor"""

import xarray as xr
import pandas as pd
import numpy as np
import unittest
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

        self.ds.temperature.attrs["u_components"] = ["u_r_temperature", "u_s_temperature"]
        self.ds.precipitation.attrs["u_components"] = ["u_r_precipitation"]

    def test_var_unc_var_names(self):
        self.assertCountEqual(self.ds.unc.var_unc_var_names("temperature"), ["u_r_temperature", "u_s_temperature"])

    def test_var_unc_var_names_none(self):
        self.assertCountEqual(self.ds.unc.var_unc_var_names("u_r_temperature"), [])

    def test_is_obs_var_true(self):
        self.assertTrue(self.ds.unc.is_obs_var("temperature"))

    def test_is_obs_var_false(self):
        self.assertFalse(self.ds.unc.is_obs_var("u_r_temperature"))

    def test_obs_vars(self):
        obs_vars = self.ds.unc.obs_vars
        self.assertEqual(type(obs_vars), xr.core.dataset.DataVariables)
        self.assertCountEqual(list(obs_vars.variables), ["temperature", "precipitation"])

    def test_unc_vars(self):
        unc_vars = self.ds.unc.unc_vars
        self.assertEqual(type(unc_vars), xr.core.dataset.DataVariables)
        self.assertCountEqual(list(unc_vars.variables), ["u_r_temperature", "u_s_temperature", "u_r_precipitation"])

    def test_is_unc_var_true(self):
        self.assertTrue(self.ds.unc.is_unc_var("u_r_temperature"))

    def test_is_unc_var_false(self):
        self.assertFalse(self.ds.unc.is_unc_var("temperature"))

    def test_var_unc_vars(self):
        unc_vars = self.ds.unc.var_unc_vars("temperature")
        self.assertEqual(type(unc_vars), xr.core.dataset.DataVariables)
        self.assertCountEqual(list(unc_vars.variables), ["u_r_temperature", "u_s_temperature"])

    def test_var_unc_vars_none(self):
        unc_vars = self.ds.unc.var_unc_vars("u_r_temperature")
        self.assertEqual(type(unc_vars), xr.core.dataset.DataVariables)
        self.assertCountEqual(list(unc_vars.variables), [])

    def test_quadsum(self):

        test_ds = xr.Dataset(
            data_vars=dict(
                a=(["x"], np.ones(3)*3),
                b=(["x"], np.ones(3)*4),
            )
        )
        c = test_ds.unc.quadsum()

        np.testing.assert_array_equal(c.values, np.ones(3)*5)

    def test_u_tot(self):
        u_tot_temperature = self.ds.unc.u_tot("temperature")
        u_tot_temperature_test = (self.ds.u_r_temperature ** 2. + self.ds.u_s_temperature ** 2.0) ** 0.5

        np.testing.assert_array_equal(u_tot_temperature.values, u_tot_temperature_test.values)

    def test_add_unc_var(self):
        u_s_precipitation = xr.DataArray(self.ds.precipitation.values, dims=["x", "y", "time"])
        self.ds.unc.add_unc_var("precipitation", ("u_s_precipitation", u_s_precipitation))
        self.assertTrue("u_s_precipitation" in self.ds.unc.var_unc_var_names("precipitation"))

        np.testing.assert_array_equal(u_s_precipitation.values, self.ds.u_s_precipitation.values)


if __name__ == "__main__":
    unittest.main()
