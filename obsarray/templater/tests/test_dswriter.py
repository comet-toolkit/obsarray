"""
Tests for DSWriter class
"""

import unittest
from unittest.mock import MagicMock
from obsarray.templater.dswriter import DSWriter


__author__ = "Sam Hunt <sam.hunt@npl.co.uk>"


# todo - write test for csv writer
# todo - write test for end to end writer use


class TestDSWriter(unittest.TestCase):
    def test__write_netcdf(self):

        ds = MagicMock()
        path = "test.nc"

        DSWriter._write_netcdf(ds, path)

        ds.to_netcdf.assert_called_once_with(
            path, encoding={}, engine="netcdf4", format="netCDF4"
        )


if __name__ == "__main__":
    unittest.main()
