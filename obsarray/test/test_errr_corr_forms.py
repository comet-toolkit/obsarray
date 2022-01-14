"""test_err_corr_forms - tests for obsarray.err_corr_forms"""

import unittest
from obsarray.err_corr_forms import BaseErrCorrForm, ErrCorrForms

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


if __name__ == "main":
    unittest.main()
