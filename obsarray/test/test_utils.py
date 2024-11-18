"""test_utils - tests for obsarray.utils"""

import unittest
import numpy as np
from obsarray import append_names, create_ds
import xarray as xr

__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"
__all__ = []

def create_test_ds(suffix):
    # define ds variables
    template = {
        "temperature" + suffix: {
            "dtype": np.float32,
            "dim": ["x" + suffix, "y" + suffix, "time" + suffix],
            "attributes": {
                "units": "K",
                "unc_comps": ["u_ran_temperature" + suffix, "u_sys_temperature" + suffix]
            }
        },
        "u_ran_temperature" + suffix: {
            "dtype": np.float32,
            "dim": ["x" + suffix, "y" + suffix, "time" + suffix],
            "attributes": {
                "units": "K",
                "err_corr": [
                    {
                        "dim": "x" + suffix,
                        "form": "random",
                        "params": [],
                        "units": []
                    },
                    {
                        "dim": "y" + suffix,
                        "form": "random",
                        "params": [],
                        "units": []
                    },
                    {
                        "dim": "time" + suffix,
                        "form": "random",
                        "params": [],
                        "units": []
                    }
                ]
            },
        },
        "u_sys_temperature" + suffix: {
            "dtype": np.float32,
            "dim": ["x" + suffix, "y" + suffix, "time" + suffix],
            "attributes": {
                "units": "K",
                "err_corr": [
                    {
                        "dim": "x" + suffix,
                        "form": "systematic",
                        "params": [],
                        "units": []
                    },
                    {
                        "dim": "y" + suffix,
                        "form": "systematic",
                        "params": [],
                        "units": []
                    },
                    {
                        "dim": "time" + suffix,
                        "form": "systematic",
                        "params": [],
                        "units": []
                    }
                ]
            }
        },
        "pressure" + suffix: {
            "dtype": np.float32,
            "dim": ["x" + suffix, "y" + suffix, "time" + suffix],
            "attributes": {
                "units": "Pa",
                "unc_comps": ["u_str_pressure" + suffix]
            }
        },
        "u_str_pressure" + suffix: {
            "dtype": np.float32,
            "dim": ["x" + suffix, "y" + suffix, "time" + suffix],
            "attributes": {
                "units": "Pa",
                "err_corr": [
                    {
                        "dim": "x" + suffix,
                        "form": "random",
                        "params": [],
                        "units": []
                    },
                    {
                        "dim": "y" + suffix,
                        "form": "err_corr_matrix",
                        "params": "err_corr_str_pressure_y",
                        "units": []
                    },
                    {
                        "dim": "time" + suffix,
                        "form": "systematic",
                        "params": [],
                        "units": []
                    }
                ]
            },
        },
        "err_corr_str_pressure_y" + suffix: {
            "dtype": np.float32,
            "dim": ["y" + suffix, "y" + suffix],
            "attributes": {"units": ""},
        },
        "n_moles" + suffix: {
            "dtype": np.float32,
            "dim": ["x" + suffix, "y" + suffix, "time" + suffix],
            "attributes": {
                "units": "",
                "unc_comps": ["u_ran_n_moles" + suffix]
            }
        },
        "u_ran_n_moles" + suffix: {
            "dtype": np.float32,
            "dim": ["x" + suffix, "y" + suffix, "time" + suffix],
            "attributes": {
                "units": "",
                "err_corr": [
                    {
                        "dim": "x" + suffix,
                        "form": "random",
                        "params": [],
                        "units": []
                    },
                    {
                        "dim": "y" + suffix,
                        "form": "random",
                        "params": [],
                        "units": []
                    },
                    {
                        "dim": "time" + suffix,
                        "form": "random",
                        "params": [],
                        "units": []
                    }
                ]
            },
        },
    }

    # define dim_size_dict to specify size of arrays
    dim_sizes = {
        "x" + suffix: 20,
        "y" + suffix: 30,
        "time" + suffix: 6
    }

    # create dataset template
    ds = create_ds(template, dim_sizes)

    # populate with example data
    ds["temperature" + suffix].values = 293 * np.ones((20, 30, 6))
    ds["u_ran_temperature" + suffix].values = 1 * np.ones((20, 30, 6))
    ds["u_sys_temperature" + suffix].values = 0.4 * np.ones((20, 30, 6))
    ds["pressure" + suffix].values = 10 ** 5 * np.ones((20, 30, 6))
    ds["u_str_pressure" + suffix].values = 10 * np.ones((20, 30, 6))
    ds["err_corr_str_pressure_y" + suffix].values = 0.5 * np.ones((30, 30)) + 0.5 * np.eye(30)
    ds["n_moles" + suffix].values = 40 * np.ones((20, 30, 6))
    ds["u_ran_n_moles" + suffix].values = 1 * np.ones((20, 30, 6))

    ds.attrs["attr" + suffix] = "val"

    return ds

class TestAppendNames(unittest.TestCase):
    def test_append_names(self):

        input_ds = create_test_ds(suffix = "")
        ds = append_names(input_ds, "_test")

        exp_ds = create_test_ds(suffix="_test")

        xr.testing.assert_identical(ds, exp_ds)

if __name__ == "__main__":
    unittest.main()
