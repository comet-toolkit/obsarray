"""test_flag_accessor - tests for obsarray.flag_accessor"""

import xarray as xr
import pandas as pd
import numpy as np
import unittest
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
            temperature_flags=(
                ["x", "y", "time"],
                np.zeros(temperature.shape, dtype=np.int8),
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

    def test_method(self):
        pass


if __name__ == "__main__":
    unittest.main()
