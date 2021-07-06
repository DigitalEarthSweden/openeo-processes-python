import unittest
import numpy as np
import openeo_processes as oeop
import pytest
import xarray as xr

@pytest.mark.usefixtures("test_data")
class LogicTester(unittest.TestCase):
    """ Tests all logic functions. """

    def test_not_(self):
        """ Tests `not_` function. """
        assert not oeop.not_(True)
        assert oeop.not_(False)
        assert oeop.not_(None) is None
        xr.testing.assert_equal(
            oeop.not_(self.test_data.xr_data_factor(False, True)),
            self.test_data.xr_data_factor(True, False))

    def test_and_(self):
        """ Tests `and_` function. """
        assert not oeop.and_(False, None)
        assert oeop.and_(True, None) is None
        assert not oeop.and_(False, False)
        assert not oeop.and_(True, False)
        assert oeop.and_(True, True)
        xr.testing.assert_equal(
            oeop.and_(self.test_data.xr_data_factor(0, 1), self.test_data.xr_data_factor(1, 1)),
            self.test_data.xr_data_factor(0, 1))
        xr.testing.assert_equal(
            oeop.and_(self.test_data.xr_data_factor(0, 1), self.test_data.xr_data_factor(np.nan, np.nan)),
            self.test_data.xr_data_factor(np.nan, np.nan))

    def test_or_(self):
        """ Tests `or_` function. """
        assert oeop.or_(False, None) is None
        assert oeop.or_(True, None)
        assert not oeop.or_(False, False)
        assert oeop.or_(True, False)
        assert oeop.or_(True, True)
        xr.testing.assert_equal(
            oeop.or_(self.test_data.xr_data_factor(0, 1), self.test_data.xr_data_factor(1, 1)),
            self.test_data.xr_data_factor(1, 1))
        xr.testing.assert_equal(
            oeop.or_(self.test_data.xr_data_factor(0, 1), self.test_data.xr_data_factor(np.nan, np.nan)),
            self.test_data.xr_data_factor(np.nan, np.nan))

    def test_xor(self):
        """ Tests `xor` function. """
        assert oeop.xor(False, None) is None
        assert oeop.xor(True, None) is None
        assert not oeop.xor(False, False)
        assert oeop.xor(True, False)
        assert not oeop.xor(True, True)
        xr.testing.assert_equal(
            oeop.xor(self.test_data.xr_data_factor(0, 1), self.test_data.xr_data_factor(1, 1)),
            self.test_data.xr_data_factor(1, 0))
        xr.testing.assert_equal(
            oeop.xor(self.test_data.xr_data_factor(0, 1), self.test_data.xr_data_factor(np.nan, np.nan)),
            self.test_data.xr_data_factor(np.nan, np.nan))

    def test_if_(self):
        """ Tests `if_` function. """
        assert oeop.if_(True, "A", "B") == "A"
        assert oeop.if_(None, "A", "B") == "B"
        assert all(oeop.if_(False, [1, 2, 3], [4, 5, 6]) == [4, 5, 6])
        assert oeop.if_(True, 123) == 123
        assert oeop.if_(False, 1) is None
        xr.testing.assert_equal(
            oeop.if_(self.test_data.xr_data_factor(0, 1), -3.5, 5),
            self.test_data.xr_data_factor(5, -3.5))
        xr.testing.assert_equal(
            oeop.if_(self.test_data.xr_data_factor(0, 1), 3),
            self.test_data.xr_data_factor(np.nan, 3))


    def test_any_(self):
        """ Tests `any_` function. """
        assert not oeop.any_([False, np.nan])
        assert oeop.any_([True, np.nan])
        assert np.isnan(oeop.any_([False, np.nan], ignore_nodata=False))
        assert oeop.any_([True, np.nan], ignore_nodata=False)
        assert oeop.any_([True, False, True, False])
        assert oeop.any_([True, False])
        assert not oeop.any_([False, False])
        assert oeop.any_([True])
        assert np.isnan(oeop.any_([np.nan], ignore_nodata=False))
        assert np.isnan(oeop.any_([]))
        xr.testing.assert_equal(
            oeop.any_(self.test_data.xr_data_factor(0, np.nan)),
            oeop.any_(self.test_data.xr_data_factor(0, 0)))
        xr.testing.assert_equal(
            oeop.any_(self.test_data.xr_data_factor(0, np.nan), ignore_nodata = False),
            oeop.any_(self.test_data.xr_data_factor(np.nan, np.nan), ignore_nodata = False))
        xr.testing.assert_equal(
            oeop.any_(self.test_data.xr_data_factor(1, np.nan), ignore_nodata=False),
            oeop.any_(self.test_data.xr_data_factor(1, 1)))

    def test_all_(self):
        """ Tests `all_` function. """
        assert not oeop.all_([False, np.nan])
        assert oeop.all_([True, np.nan])
        assert not oeop.all_([False, np.nan], ignore_nodata=False)
        assert np.isnan(oeop.all_([True, np.nan], ignore_nodata=False))
        assert not oeop.all_([True, False, True, False])
        assert not oeop.all_([True, False])
        assert oeop.all_([True, True])
        assert oeop.all_([True])
        assert np.isnan(oeop.all_([np.nan], ignore_nodata=False))
        assert np.isnan(oeop.all_([]))
        xr.testing.assert_equal(
            oeop.all_(self.test_data.xr_data_factor(0, 1)),
            oeop.all_(self.test_data.xr_data_factor(0, 0)))
        xr.testing.assert_equal(
            oeop.all_(self.test_data.xr_data_factor(1, np.nan), ignore_nodata=False),
            oeop.all_(self.test_data.xr_data_factor(np.nan, np.nan), ignore_nodata=False))
        xr.testing.assert_equal(
            oeop.all_(self.test_data.xr_data_factor(0, np.nan), ignore_nodata=False),
            oeop.all_(self.test_data.xr_data_factor(0, 0)))

if __name__ == '__main__':
    unittest.main()
