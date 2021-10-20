"""test_accessor - tests for obsarray.accessor"""

import xarray as xr
import pandas as pd
import numpy as np
import unittest
import obsarray

__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"
__all__ = []

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

ds = xr.Dataset(
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

ds.temperature.attrs["u_components"] = ["u_r_temperature", "u_s_temperature"]
ds.precipitation.attrs["u_components"] = ["u_r_precipitation"]


class TestUncAccessor(unittest.TestCase):
    def test_measurement_variables(self):
        self.assertCountEqual(ds.unc.measured_variables, ["temperature", "precipitation"])

    def test_uncertainty_variables(self):
        self.assertCountEqual(ds.unc.uncertainty_variables, ["u_r_temperature", "u_s_temperature", "u_r_precipitation"])


if __name__ == "__main__":
    unittest.main()
