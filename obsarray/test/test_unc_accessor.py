"""test_unc_accessor - tests for obsarray.unc_accessor"""

from copy import deepcopy
import xarray as xr
import pandas as pd
import numpy as np
import unittest
import unittest.mock as mock
from unittest.mock import patch
import obsarray


__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"
__all__ = []


simple_ds = xr.Dataset(
    {"a": (["x", "y"], np.full((2, 2), 3)), "b": (["x", "y"], np.full((2, 2), 4))}
).data_vars


def simple_build_matrix(sli):
    return np.eye(12)


mock_err_corr_form = mock.MagicMock()
mock_err_corr_form.return_value.build_matrix = simple_build_matrix


def compare_err_corr_form(self, form, exp_form):
    self.assertEqual(form.form, exp_form.form)
    self.assertCountEqual(form.params, exp_form.params)
    self.assertCountEqual(form.units, exp_form.units)
    self.assertCountEqual(form._unc_var_name, exp_form._unc_var_name)


def create_ds():
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
            temperature=(["x", "y", "time"], temperature, {"units": "K"}),
        ),
        coords=dict(
            lon=(["x", "y"], lon),
            lat=(["x", "y"], lat),
            time=time,
            reference_time=reference_time,
        ),
        attrs=dict(description="Weather related data."),
    )

    ds.unc["temperature"]["u_ran_temperature"] = (
        ["x", "y", "time"],
        temperature * 0.05,
        {"units": "K", "pdf_shape": "gaussian"},
    )

    ds.unc["temperature"]["u_sys_temperature"] = (
        ["x", "y", "time"],
        temperature * 0.03,
        {
            "units": "K",
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
            ],
            "pdf_shape": "gaussian",
        },
    )

    ds.unc["temperature"]["u_str_temperature"] = (
        ["x", "y", "time"],
        temperature * 0.1,
        {
            "units": "K",
            "err_corr": [
                {
                    "dim": "x",
                    "form": "err_corr_matrix",
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
            ],
            "pdf_shape": "gaussian",
        },
    )

    return ds


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
                u_r_temperature=(
                    ["x", "y", "time"],
                    u_r_temperature,
                    {
                        "err_corr_1_dim": "x",
                        "err_corr_1_form": "random",
                        "err_corr_1_params": [],
                        "err_corr_1_units": [],
                        "err_corr_2_dim": "y",
                        "err_corr_2_form": "random",
                        "err_corr_2_params": [],
                        "err_corr_2_units": [],
                        "err_corr_3_dim": "time",
                        "err_corr_3_form": "random",
                        "err_corr_3_params": [],
                        "err_corr_3_units": [],
                    },
                ),
                u_s_temperature=(
                    ["x", "y", "time"],
                    u_s_temperature,
                    {
                        "err_corr_1_dim": "x",
                        "err_corr_1_form": "systematic",
                        "err_corr_1_params": [],
                        "err_corr_1_units": [],
                        "err_corr_2_dim": "y",
                        "err_corr_2_form": "systematic",
                        "err_corr_2_params": [],
                        "err_corr_2_units": [],
                        "err_corr_3_dim": "time",
                        "err_corr_3_form": "systematic",
                        "err_corr_3_params": [],
                        "err_corr_3_units": [],
                    },
                ),
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

    def test___getitem__(self):
        self.assertIsInstance(
            self.ds.unc["temperature"], obsarray.unc_accessor.VariableUncertainty
        )
        self.assertEqual(self.ds.unc["temperature"]._var_name, "temperature")

    def test___len__(self):
        self.assertEqual(len(self.ds.unc), 2)

    def test___iter__(self):

        var_names = []
        for unc in self.ds.unc:
            self.assertIsInstance(unc, obsarray.unc_accessor.VariableUncertainty)
            var_names.append(unc._var_name)

        self.assertCountEqual(var_names, ["temperature", "precipitation"])

    def test_keys(self):
        self.assertCountEqual(self.ds.unc.keys(), ["temperature", "precipitation"])

    def test__var_unc_var_names(self):
        self.assertCountEqual(
            self.ds.unc._var_unc_var_names("temperature"),
            ["u_r_temperature", "u_s_temperature"],
        )

    def test__var_unc_var_names_none(self):
        self.assertCountEqual(self.ds.unc._var_unc_var_names("u_r_temperature"), [])

    def test__var_unc_var_names_random(self):
        self.assertCountEqual(
            self.ds.unc._var_unc_var_names("temperature", "random"),
            ["u_r_temperature"],
        )

    def test__var_unc_var_names_structured(self):
        self.assertCountEqual(
            self.ds.unc._var_unc_var_names("temperature", "structured"),
            [],
        )

    def test__var_unc_var_names_systematic(self):
        self.assertCountEqual(
            self.ds.unc._var_unc_var_names("temperature", "systematic"),
            ["u_s_temperature"],
        )

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
        self.assertIsNone(self.ds.unc._var_unc_vars("u_r_temperature"))

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

        self.assertTrue(("u_s_precipitation" in self.ds))

    def test__remove_unc_var(self):
        self.ds.unc._remove_unc_var("temperature", "u_s_temperature")

        self.assertTrue("u_s_temperature" not in self.ds)
        self.assertTrue(
            "u_s_temperature" not in self.ds["precipitation"].attrs["unc_comps"]
        )


class TestVariableUncertainty(unittest.TestCase):
    def setUp(self):
        self.ds = create_ds()

    def test___getitem__(self):
        self.assertIsInstance(
            self.ds.unc["temperature"]["u_ran_temperature"],
            obsarray.unc_accessor.Uncertainty,
        )
        self.assertEqual(
            self.ds.unc["temperature"]["u_ran_temperature"]._unc_var_name,
            "u_ran_temperature",
        )

    def test___len__(self):
        self.assertEqual(len(self.ds.unc["temperature"]), 3)

    def test___iter__(self):

        var_names = []
        for unc in self.ds.unc["temperature"]:
            self.assertIsInstance(unc, obsarray.unc_accessor.Uncertainty)
            var_names.append(unc._unc_var_name)

        self.assertCountEqual(
            var_names, ["u_ran_temperature", "u_sys_temperature", "u_str_temperature"]
        )

    def test_keys(self):
        self.assertCountEqual(
            self.ds.unc["temperature"].keys(),
            ["u_ran_temperature", "u_sys_temperature", "u_str_temperature"],
        )

    @patch("obsarray.unc_accessor.UncAccessor._var_unc_vars")
    def test_comps(self, mock_method):
        comps = self.ds.unc["temperature"][:, :, 0].comps
        mock_method.assert_called_once_with(
            "temperature", (slice(None), slice(None), 0)
        )

        self.assertEqual(comps, mock_method.return_value)

    @patch("obsarray.unc_accessor.UncAccessor._var_unc_vars")
    def test_random_comps(self, mock_method):
        comps = self.ds.unc["temperature"][:, :, 0].random_comps
        mock_method.assert_called_once_with(
            "temperature", (slice(None), slice(None), 0), unc_type="random"
        )

        self.assertEqual(comps, mock_method.return_value)

    @patch("obsarray.unc_accessor.UncAccessor._var_unc_vars")
    def test_structured_comps(self, mock_method):
        comps = self.ds.unc["temperature"][:, :, 0].structured_comps
        mock_method.assert_called_once_with(
            "temperature", (slice(None), slice(None), 0), unc_type="structured"
        )

        self.assertEqual(comps, mock_method.return_value)

    @patch("obsarray.unc_accessor.UncAccessor._var_unc_vars")
    def test_systematic_comps(self, mock_method):
        comps = self.ds.unc["temperature"][:, :, 0].systematic_comps
        mock_method.assert_called_once_with(
            "temperature", (slice(None), slice(None), 0), unc_type="systematic"
        )

        self.assertEqual(comps, mock_method.return_value)

    def test__quadsum_unc(self):

        self.ds["u_ran_temperature"].values[:] = 3.0
        self.ds["u_str_temperature"].values[:] = (
            4.0 / self.ds["temperature"].values * 100.0
        )
        self.ds["u_str_temperature"].attrs["units"] = "%"

        u_tot = self.ds.unc["temperature"][:, :, 0]._quadsum_unc(
            ["u_ran_temperature", "u_str_temperature"]
        )
        exp_u_tot = deepcopy(self.ds["temperature"])
        exp_u_tot.values[:] = 5.0

        xr.testing.assert_allclose(u_tot, exp_u_tot[:, :, 0])

    def test__quadsum_unc_None(self):
        self.assertIsNone(self.ds.unc["temperature"]._quadsum_unc([]))

    @patch("obsarray.unc_accessor.VariableUncertainty._quadsum_unc")
    def test_total_unc(self, mock):
        self.ds.unc["temperature"][:, :, 0].total_unc()
        mock.assert_called_once()
        self.assertCountEqual(
            mock.mock_calls[0][1][0],
            ["u_ran_temperature", "u_str_temperature", "u_sys_temperature"],
        )

    @patch("obsarray.unc_accessor.VariableUncertainty._quadsum_unc")
    def test_random_unc(self, mock):
        self.ds.unc["temperature"][:, :, 0].random_unc()
        mock.assert_called_once_with(["u_ran_temperature"])

    @patch("obsarray.unc_accessor.VariableUncertainty._quadsum_unc")
    def test_structured_unc(self, mock):
        self.ds.unc["temperature"][:, :, 0].structured_unc()
        mock.assert_called_once_with(["u_str_temperature"])

    @patch("obsarray.unc_accessor.VariableUncertainty._quadsum_unc")
    def test_systematic_unc(self, mock):
        self.ds.unc["temperature"][:, :, 0].systematic_unc()
        mock.assert_called_once_with(["u_sys_temperature"])

    @patch(
        "obsarray.unc_accessor.Uncertainty.err_corr_matrix",
        return_value=xr.DataArray(np.ones((12, 12)), dims=["x.y.time", "x.y.time"]),
    )
    def test_total_err_corr_matrix(self, mock_err_corr_matrix):
        pass
        # tercm = self.ds.unc["temperature"].total_err_corr_matrix()

    def test_structured_err_corr_matrix(self):
        pass

    def test_total_err_cov_matrix(self):
        pass

    def test_structured_err_cov_matrix(self):
        pass


class TestUncertainty(unittest.TestCase):
    def setUp(self):
        self.ds = create_ds()

    @patch("obsarray.unc_accessor.Uncertainty.expand_sli", return_value="slice")
    def test___getitem__(self, m):
        self.assertEqual(
            self.ds.unc["temperature"]["u_ran_temperature"]["in_slice"]._sli, "slice"
        )

        m.assert_called_with("in_slice")

    def test_expand_slice_1d_full(self):
        self.ds["new"] = (["time"], np.ones(3), {})
        self.ds.unc["new"]["u_new"] = (["time"], np.ones(3), {})
        sli = self.ds.unc["new"]["u_new"]._expand_sli((1))
        self.assertEqual((1,), sli)

    def test_expand_slice_1d_None(self):
        self.ds["new"] = (["time"], np.ones(3), {})
        self.ds.unc["new"]["u_new"] = (["time"], np.ones(3), {})
        sli = self.ds.unc["temperature"]["u_ran_temperature"]._expand_sli()
        self.assertEqual((slice(None),), sli)

    def test_expand_slice_full(self):
        sli = self.ds.unc["temperature"]["u_ran_temperature"].expand_sli((1, 1, 1))
        self.assertEqual((1, 1, 1), sli)

    def test_expand_slice_None(self):
        sli = self.ds.unc["temperature"]["u_ran_temperature"].expand_sli()
        self.assertEqual((slice(None), slice(None), slice(None)), sli)

    def test_expand_slice_first(self):
        sli = self.ds.unc["temperature"]["u_ran_temperature"].expand_sli((0,))
        self.assertEqual((0, slice(None), slice(None)), sli)

    def test_err_corr(self):

        expected_err_corr = [
            (
                "y",
                obsarray.err_corr.RandomCorrelation(
                    self.ds, "u_ran_temperature", ["y"], [], []
                ),
            ),
            (
                "time",
                obsarray.err_corr.RandomCorrelation(
                    self.ds, "u_ran_temperature", ["time"], [], []
                ),
            ),
            (
                "x",
                obsarray.err_corr.RandomCorrelation(
                    self.ds, "u_ran_temperature", "x", [], []
                ),
            ),
        ]

        err_corr = self.ds.unc["temperature"]["u_ran_temperature"].err_corr

        for dim_tp, exp_dim_tp in zip(sorted(err_corr), sorted(expected_err_corr)):
            self.assertEqual(dim_tp[0], exp_dim_tp[0])
            compare_err_corr_form(self, dim_tp[1], exp_dim_tp[1])

    def test_err_corr_slice(self):

        expected_err_corr = [
            (
                "y",
                obsarray.err_corr.RandomCorrelation(
                    self.ds, "u_ran_temperature", ["y"], [], []
                ),
            ),
            (
                "time",
                obsarray.err_corr.RandomCorrelation(
                    self.ds, "u_ran_temperature", ["time"], [], []
                ),
            ),
        ]

        err_corr = self.ds.unc["temperature"]["u_ran_temperature"][0, :, :].err_corr

        for dim_tp, exp_dim_tp in zip(sorted(err_corr), sorted(expected_err_corr)):
            self.assertEqual(dim_tp[0], exp_dim_tp[0])
            compare_err_corr_form(self, dim_tp[1], exp_dim_tp[1])

    def test_units_K(self):

        self.assertEqual(self.ds.unc["temperature"]["u_ran_temperature"].units, "K")

    def test_units_None(self):

        self.ds["u_str_temperature"].attrs.pop("units")

        self.assertIsNone(
            self.ds.unc["temperature"]["u_str_temperature"].units,
        )

    def test_var_units_K(self):

        self.assertEqual(self.ds.unc["temperature"]["u_ran_temperature"].var_units, "K")

    def test_var_units_None(self):

        self.ds["temperature"].attrs.pop("units")

        self.assertIsNone(
            self.ds.unc["temperature"]["u_ran_temperature"].var_units,
        )

    def test_value(self):

        xr.testing.assert_equal(
            self.ds["u_ran_temperature"],
            self.ds.unc["temperature"]["u_ran_temperature"].value,
        )

    def test_value_slice(self):

        xr.testing.assert_equal(
            self.ds["u_ran_temperature"][:, 0, :],
            self.ds.unc["temperature"]["u_ran_temperature"][:, 0, :].value,
        )

    def test_var_value(self):

        xr.testing.assert_equal(
            self.ds["temperature"],
            self.ds.unc["temperature"]["u_ran_temperature"].var_value,
        )

    def test_var_value_slice(self):

        xr.testing.assert_equal(
            self.ds["temperature"][:, 0, :],
            self.ds.unc["temperature"]["u_ran_temperature"][:, 0, :].var_value,
        )

    def test_abs_value_percentage(self):
        self.ds["u_str_temperature"].attrs["units"] = "%"
        self.ds["u_str_temperature"].values[:] = 10.0

        exp_da = deepcopy(self.ds["u_str_temperature"])
        exp_da.values = self.ds["temperature"] / 10.0

        xr.testing.assert_allclose(
            exp_da,
            self.ds.unc["temperature"]["u_str_temperature"].abs_value,
        )

    def test_abs_value_sameunits(self):

        xr.testing.assert_equal(
            self.ds.unc["temperature"]["u_str_temperature"].value,
            self.ds.unc["temperature"]["u_str_temperature"].abs_value,
        )

    def test_abs_value_diffunits(self):

        self.ds["u_str_temperature"].attrs["units"] = "mK"

        def unitcall(ds):
            ds.unc["temperature"]["u_str_temperature"].abs_value

        self.assertRaises(ValueError, unitcall, self.ds)

    def test_abs_value_uncnounits(self):

        del self.ds["u_str_temperature"].attrs["units"]

        xr.testing.assert_equal(
            self.ds.unc["temperature"]["u_str_temperature"].value,
            self.ds.unc["temperature"]["u_str_temperature"].abs_value,
        )

    def test_pdf_shape(self):
        self.assertEqual(
            self.ds.unc["temperature"]["u_ran_temperature"].pdf_shape, "gaussian"
        )

    def test_is_random_true(self):

        self.assertTrue(self.ds.unc["temperature"]["u_ran_temperature"].is_random)

    def test_is_random_false(self):

        self.assertFalse(self.ds.unc["temperature"]["u_sys_temperature"].is_random)

    def test_is_structured_true(self):

        self.assertTrue(self.ds.unc["temperature"]["u_str_temperature"].is_structured)

    def test_is_structured_false_ran(self):

        self.assertFalse(self.ds.unc["temperature"]["u_ran_temperature"].is_structured)

    def test_is_structured_false_sys(self):

        self.assertFalse(self.ds.unc["temperature"]["u_sys_temperature"].is_structured)

    def test_is_systematic_true(self):

        self.assertTrue(self.ds.unc["temperature"]["u_sys_temperature"].is_systematic)

    def test_is_systematic_false(self):

        self.assertFalse(self.ds.unc["temperature"]["u_ran_temperature"].is_systematic)

    @patch("obsarray.unc_accessor.err_corr_forms", {"random": mock_err_corr_form})
    def test_err_corr_matrix_3dvar(self):
        erc = self.ds.unc["temperature"]["u_ran_temperature"].err_corr_matrix()

        exp_erc = xr.DataArray(np.eye(12), dims=["x.y.time", "x.y.time"])
        xr.testing.assert_equal(erc, exp_erc)

    @patch("obsarray.unc_accessor.Uncertainty.value")
    @patch("obsarray.unc_accessor.Uncertainty.err_corr_matrix")
    @patch("obsarray.unc_accessor.convert_corr_to_cov", return_value=np.ones((12, 12)))
    def test_err_cov_matrix(self, mock_cr2cv, mock_ecrm, mock_value):
        ecm = self.ds.unc["temperature"]["u_ran_temperature"].err_cov_matrix()

        mock_cr2cv.assert_called_once_with(
            mock_ecrm.return_value.values, mock_value.values
        )

        exp_ecm = xr.DataArray(np.ones((12, 12)), dims=["x.y.time", "x.y.time"])
        xr.testing.assert_equal(ecm, exp_ecm)


if __name__ == "__main__":
    unittest.main()
