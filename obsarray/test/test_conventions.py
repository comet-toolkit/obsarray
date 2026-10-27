"""test_conventions - tests for obsarray.conventions"""

import unittest
import warnings
import numpy as np
import xarray as xr
from obsarray import UNCConventionsWarning, create_ds, get_options, set_options
from obsarray import conventions
from obsarray.options import OPTIONS

__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"
__all__ = []


def create_obs_ds():
    return xr.Dataset({"temperature": (["x"], np.ones(3), {"units": "K"})})


class TestUNCConventionsWarning(unittest.TestCase):
    def setUp(self):
        # each test starts as a fresh session that hasn't warned yet
        self.original_warned = conventions._warned
        self.original_options = get_options()
        conventions._warned = False

        self.ds = create_obs_ds()

    def tearDown(self):
        conventions._warned = self.original_warned
        OPTIONS.update(self.original_options)

    def add_unc_var(self, unc_var="u_noise"):
        self.ds.unc["temperature"][unc_var] = (["x"], np.ones(3) * 0.1, {"units": "K"})

    def assertNoUNCConventionsWarning(self, func, *args, **kwargs):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            func(*args, **kwargs)

        self.assertFalse(
            [w for w in caught if issubclass(w.category, UNCConventionsWarning)]
        )

    def test_is_future_warning(self):
        # FutureWarning is shown to end users by default, unlike DeprecationWarning
        self.assertTrue(issubclass(UNCConventionsWarning, FutureWarning))

    def test_warns_on_adding_unc_var(self):
        with self.assertWarns(UNCConventionsWarning):
            self.add_unc_var()

    def test_warns_on_adding_unc_var_dataarray(self):
        unc = xr.DataArray(np.ones(3) * 0.1, dims=["x"], attrs={"units": "K"})

        with self.assertWarns(UNCConventionsWarning):
            self.ds.unc["temperature"]["u_noise"] = unc

    def test_warns_on_template_unc_var(self):
        template = {
            "temperature": {
                "dim": ["x"],
                "dtype": np.float32,
                "attributes": {"units": "K", "unc_comps": ["u_temperature"]},
            },
            "u_temperature": {
                "dim": ["x"],
                "dtype": np.float32,
                "attributes": {
                    "units": "K",
                    "err_corr": [{"dim": "x", "form": "random", "params": []}],
                },
            },
        }

        with self.assertWarns(UNCConventionsWarning):
            create_ds(template, {"x": 3})

    def test_warns_once_per_session(self):
        with self.assertWarns(UNCConventionsWarning):
            self.add_unc_var("u_noise")

        self.assertNoUNCConventionsWarning(self.add_unc_var, "u_calibration")

    def test_no_warning_when_conventions_set(self):
        set_options(unc_conventions="v0.1")

        self.assertNoUNCConventionsWarning(self.add_unc_var)

    def test_no_warning_within_set_options_block(self):
        with set_options(unc_conventions="v0.1"):
            self.assertNoUNCConventionsWarning(self.add_unc_var, "u_noise")

        # not silenced for good - the next write outside the block still warns
        with self.assertWarns(UNCConventionsWarning):
            self.add_unc_var("u_calibration")

    def test_no_warning_on_read(self):
        with set_options(unc_conventions="v0.1"):
            self.add_unc_var()

        self.assertNoUNCConventionsWarning(
            lambda: self.ds.unc["temperature"]["u_noise"].value
        )

    def test_warning_points_at_calling_code(self):
        user_code = compile(
            'ds.unc["temperature"]["u_noise"] = (["x"], u, {"units": "K"})',
            "user_script.py",
            "exec",
        )

        with self.assertWarns(UNCConventionsWarning) as cm:
            exec(user_code, {"ds": self.ds, "u": np.ones(3) * 0.1})

        self.assertEqual(cm.filename, "user_script.py")

    def test_message_explains_how_to_silence(self):
        with self.assertWarns(UNCConventionsWarning) as cm:
            self.add_unc_var()

        message = str(cm.warning)
        self.assertIn('set_options(unc_conventions="v0.1")', message)
        self.assertIn(conventions.UNC_CONVENTIONS_DOCS_URL, message)


if __name__ == "__main__":
    unittest.main()
