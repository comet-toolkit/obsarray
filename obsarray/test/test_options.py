"""test_options - tests for obsarray.options"""

import unittest
from obsarray import set_options, get_options
from obsarray.options import OPTIONS

__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"
__all__ = []


class TestSetOptions(unittest.TestCase):
    def setUp(self):
        self.original_options = get_options()

    def tearDown(self):
        OPTIONS.update(self.original_options)

    def test_default(self):
        self.assertIsNone(get_options()["unc_conventions"])

    def test_set_globally(self):
        set_options(unc_conventions="v0.1")
        self.assertEqual(get_options()["unc_conventions"], "v0.1")

    def test_context_manager_restores(self):
        with set_options(unc_conventions="v0.1"):
            self.assertEqual(get_options()["unc_conventions"], "v0.1")

        self.assertIsNone(get_options()["unc_conventions"])

    def test_context_manager_restores_on_error(self):
        with self.assertRaises(RuntimeError):
            with set_options(unc_conventions="v0.1"):
                raise RuntimeError

        self.assertIsNone(get_options()["unc_conventions"])

    def test_invalid_option(self):
        self.assertRaises(ValueError, set_options, not_an_option="v0.1")

    def test_invalid_value(self):
        self.assertRaises(ValueError, set_options, unc_conventions="v0.2")
        self.assertIsNone(get_options()["unc_conventions"])

    def test_get_options_returns_copy(self):
        get_options()["unc_conventions"] = "v0.1"
        self.assertIsNone(get_options()["unc_conventions"])


if __name__ == "__main__":
    unittest.main()
