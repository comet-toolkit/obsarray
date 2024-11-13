"""test_err_corr_forms - tests for obsarray.err_corr_forms"""

import unittest
import numpy as np
import obsarray
import xarray as xr


__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"
__all__ = []

from obsarray.concat import obs_concat


def create_test_ds():
    c1a = np.ones((4, 3))
    c2a = np.ones((4, 3))

    c1b = np.ones((7, 5, 3))
    c2b = np.ones((7, 5, 3))

    d1a = np.ones((4, 3)) * 1
    d2a = np.ones((4, 3)) * 2
    s1a = np.ones((4, 3)) * 3
    s2a = np.ones((4, 3)) * 4
    s3a = np.ones((4, 3)) * 5

    d1b = np.ones((7, 5, 3)) * 1

    d1a_attrs = {"units": "test_units", "geometry": "a", "measurand": "d", 'm': 10}
    d2a_attrs = {"units": "test_units", "geometry": "a", "measurand": "d", 'm': 11}
    s1a_attrs = {"units": "test_units", "geometry": "a", "measurand": "s", 'm': 12}
    s2a_attrs = {"units": "test_units", "geometry": "a", "measurand": "s", 'm': 4}
    s3a_attrs = {"units": "test_units", "geometry": "a", "measurand": "s", 'm': 5}
    d1b_attrs = {"units": "test_units", "geometry": "b", "measurand": "d"}

    ds = xr.Dataset(
        {
            "d1a": (["xa", "ya"], d1a, d1a_attrs),
            "d2a": (["xa", "ya"], d2a, d2a_attrs),
            "s1a": (["xa", "ya"], s1a, s1a_attrs),
            "s2a": (["xa", "ya"], s2a, s2a_attrs),
            "s3a": (["xa", "ya"], s3a, s3a_attrs),
            "d1b": (["xb", "yb", "zb"], d1b, d1b_attrs),
        },
        coords={
            "c1a": (["xa", "ya"], c1a),
            "c2a": (["xa", "ya"], c2a),
            "c1b": (["xb", "yb", "zb"], c1b),
            "c2b": (["xb", "yb", "zb"], c2b),
        },
        attrs={
            "history": "test_history",
            "meas_vars": ["d1a", "d2a", "s1a", "s2a", "s3a", "d1b"],
        },
    )

    for var in ["d1a", "d2a"]:
        ds.unc[var]["u_r_" + var] = (["xa", "ya"], ds[var].values, {})

        err_corr_def = [
            {
                "dim": ["xa", "ya"],
                "form": "systematic",
                "params": [],
                "units": []
            }
        ]

        ds.unc[var]["u_s_" + var] = (["xa", "ya"], ds[var].values, {"err_corr": err_corr_def})

    return ds

class TestConcat(unittest.TestCase):

    def test_concat_combine_unc_concat(self):
        ds = create_test_ds()

        concat_vars, concat_unc_vars = obs_concat([ds["d1a"], ds["d2a"]], "new_dim", ds, "concat")

        # Create expected concatenated data array
        data = np.array([[[1., 1., 1.],
                          [1., 1., 1.],
                          [1., 1., 1.],
                          [1., 1., 1.]],
                         [[2., 2., 2.],
                          [2., 2., 2.],
                          [2., 2., 2.],
                          [2., 2., 2.]]])

        coord_data = np.ones((4, 3))  # coords

        exp_concat_vars = xr.DataArray(
            data,
            dims=("new_dim", "xa", "ya"),
            coords={
                "c1a": (("xa", "ya"), coord_data),
                "c2a": (("xa", "ya"), coord_data),
            },
            name="d1a",
            attrs={
                "units": "test_units",
                "geometry": "a",
                "measurand": "d",
                "m": 10,
                "unc_comps": ["u_r_d1a", "u_s_d1a"]
            }
        )

        # Tests
        xr.testing.assert_equal(concat_vars, exp_concat_vars)
        self.assertIsNone(concat_unc_vars)

if __name__ == "__main__":
    pass
