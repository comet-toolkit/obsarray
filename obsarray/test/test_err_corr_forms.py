"""test_err_corr_forms - tests for obsarray.err_corr_forms"""

import unittest
import numpy as np
from obsarray.err_corr import (
    BaseErrCorrForm,
    ErrCorrForms,
    RandomCorrelation,
    SystematicCorrelation,
)
from obsarray.test.test_unc_accessor import create_ds

__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"
__all__ = []


class TestErrCorrForms(unittest.TestCase):
    def setUp(self):
        class ErrCorrFormDef(BaseErrCorrForm):
            def form(self):
                pass

        self.ErrCorrFormDef = ErrCorrFormDef

    def test___setitem__(self):
        err_corr_forms = ErrCorrForms()
        err_corr_forms["test"] = self.ErrCorrFormDef

        self.assertCountEqual(list(err_corr_forms._forms.keys()), ["test"])

    def test___setitem___not_valid_cls(self):
        class InvalidErrCorrFormDef:
            pass

        err_corr_forms = ErrCorrForms()
        self.assertRaises(
            TypeError, err_corr_forms.__setitem__, "test", InvalidErrCorrFormDef
        )

    def test___getitem__(self):
        err_corr_forms = ErrCorrForms()
        err_corr_forms._forms["test"] = "test"
        self.assertEqual("test", err_corr_forms["test"])

    def test___delitem__(self):
        err_corr_forms = ErrCorrForms()
        err_corr_forms._forms["test"] = "test"

        del err_corr_forms._forms["test"]

        self.assertTrue("test" not in err_corr_forms._forms)

    def test_keys(self):
        err_corr_forms = ErrCorrForms()
        err_corr_forms._forms["test"] = "test"

        self.assertCountEqual(["test"], err_corr_forms.keys())


class TestBaseErrCorrForm(unittest.TestCase):
    def setUp(self) -> None:
        self.ds = create_ds()

        class BasicErrCorrForm(BaseErrCorrForm):
            form = "basic"

            def build_matrix(self, sli):
                return None

        self.BasicErrCorrForm = BasicErrCorrForm

    def test_slice_full_cov_full(self):
        basicerrcorr = self.BasicErrCorrForm(
            self.ds, "u_ran_temperature", ["x"], [], []
        )

        full_matrix = np.arange(144).reshape((12, 12))
        slice_matrix = basicerrcorr.slice_full_cov(
            full_matrix, (slice(None), slice(None), slice(None))
        )

        np.testing.assert_equal(full_matrix, slice_matrix)

    def test_slice_full_cov_slice(self):
        basicerrcorr = self.BasicErrCorrForm(
            self.ds, "u_ran_temperature", ["x"], [], []
        )

        full_matrix = np.arange(144).reshape((12, 12))
        slice_matrix = basicerrcorr.slice_full_cov(
            full_matrix, (slice(None), slice(None), 0)
        )

        exp_slice_matrix = np.array(
            [[0, 3, 6, 9], [36, 39, 42, 45], [72, 75, 78, 81], [108, 111, 114, 117]]
        )

        np.testing.assert_equal(slice_matrix, exp_slice_matrix)


class TestRandomUnc(unittest.TestCase):
    def setUp(self) -> None:
        self.ds = create_ds()

    def test_build_matrix_1stdim(self):
        rc = RandomCorrelation(self.ds, "u_ran_temperature", ["x"], [], [])

        ecrm = rc.build_matrix((slice(None), slice(None), slice(None)))

        np.testing.assert_equal(ecrm, np.eye(12))

    def test_build_matrix_2nddim(self):
        rc = RandomCorrelation(self.ds, "u_ran_temperature", ["y"], [], [])

        ecrm = rc.build_matrix((slice(None), slice(None), slice(None)))

        np.testing.assert_equal(ecrm, np.eye(12))


class TestSystematicUnc(unittest.TestCase):
    def setUp(self) -> None:
        self.ds = create_ds()

    def build_matrix_1stdim(self):
        rc = SystematicCorrelation(self.ds, "u_sys_temperature", ["x"], [], [])

        ecrm = rc.build_matrix((slice(None), slice(None), slice(None)))
        # np.testing.assert_equal(ecrm, np.ones((12, 12)))

        return ecrm

    def build_matrix_2nddim(self):
        rc = SystematicCorrelation(self.ds, "u_sys_temperature", ["y"], [], [])

        ecrm = rc.build_matrix((slice(None), slice(None), slice(None)))
        # np.testing.assert_equal(ecrm, np.ones((12, 12)))

        return ecrm

    def build_matrix_3ddim(self):
        rc = SystematicCorrelation(self.ds, "u_sys_temperature", ["time"], [], [])

        ecrm = rc.build_matrix((slice(None), slice(None), slice(None)))
        # np.testing.assert_equal(ecrm, np.ones((12, 12)))

        return ecrm

    def test_build_matrix(self):
        x = self.build_matrix_1stdim()
        y = self.build_matrix_2nddim()
        time = self.build_matrix_3ddim()
        np.testing.assert_equal((x.dot(y)).dot(time), np.ones((12, 12)))


if __name__ == "main":
    unittest.main()
