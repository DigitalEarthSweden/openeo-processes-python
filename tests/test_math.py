"""
Most tests are in alignment with:
https://openeo.org/documentation/1.0/processes.html
"""

import unittest
import numpy as np
import pytest
from copy import deepcopy
import openeo_processes as oeop
import xarray as xr
import scipy

@pytest.mark.usefixtures("test_data")
class MathTester(unittest.TestCase):
    """ Tests all math functions. """

    def test_e(self):
        """ Tests `e` function. """
        assert oeop.e() == np.e

    def test_pi(self):
        """ Tests `pi` function. """
        assert oeop.pi() == np.pi

    def test_floor(self):
        """ Tests `floor` function. """
        assert oeop.floor(0) == 0
        assert oeop.floor(3.5) == 3
        assert oeop.floor(-0.4) == -1
        assert oeop.floor(-3.5) == -4
        xr.testing.assert_equal(oeop.floor(self.test_data.xr_data_factor(3.5, -3.5)), self.test_data.xr_data_factor(3, -4))

    def test_ceil(self):
        """ Tests `ceil` function. """
        assert oeop.ceil(0) == 0
        assert oeop.ceil(3.5) == 4
        assert oeop.ceil(-0.4) == 0
        assert oeop.ceil(-3.5) == -3
        xr.testing.assert_equal(
            oeop.ceil(self.test_data.xr_data_factor(3.5, -3.5)), self.test_data.xr_data_factor(4, -3))

    def test_int(self):
        """ Tests `int` function. """
        assert oeop.int(0) == 0
        assert oeop.int(3.5) == 3
        assert oeop.int(-0.4) == 0
        assert oeop.int(-3.5) == -3
        xr.testing.assert_equal(
            oeop.int(self.test_data.xr_data_factor(3.5, -3.5)), self.test_data.xr_data_factor(3, -3))

    def test_round(self):
        """ Tests `round` function. """
        assert oeop.round(0) == 0
        assert oeop.round(3.56, p=1) == 3.6
        assert oeop.round(-0.4444444, p=2) == -0.44
        assert oeop.round(-2.5) == -2
        assert oeop.round(-3.5) == -4
        assert oeop.round(1234.5, p=-2) == 1200
        xr.testing.assert_equal(
            oeop.round(self.test_data.xr_data_factor(-2.5, -3.5)), self.test_data.xr_data_factor(-2, -4))

    def test_exp(self):
        """ Tests `exp` function. """
        assert oeop.exp(0) == 1
        assert np.isnan(oeop.exp(np.nan))
        xr.testing.assert_equal(
            oeop.exp(self.test_data.xr_data_factor(0, np.nan)), self.test_data.xr_data_factor(1, np.nan))

    def test_log(self):
        """ Tests `log` function. """
        assert oeop.log(10, 10) == 1
        assert oeop.log(2, 2) == 1
        assert oeop.log(4, 2) == 2
        assert oeop.log(1, 16) == 0
        xr.testing.assert_equal(
            oeop.log(self.test_data.xr_data_factor(10, 10), 10), self.test_data.xr_data_factor(1, 1))

    def test_ln(self):
        """ Tests `ln` function. """
        assert oeop.ln(oeop.e()) == 1
        assert oeop.ln(1) == 0
        xr.testing.assert_equal(
            oeop.ln(self.test_data.xr_data_factor(oeop.e(), 1)), self.test_data.xr_data_factor(1, 0))

    def test_cos(self):
        """ Tests `cos` function. """
        assert oeop.cos(0) == 1
        xr.testing.assert_equal(
            oeop.cos(self.test_data.xr_data_factor(oeop.pi(), 0)), self.test_data.xr_data_factor(-1, 1))

    def test_arccos(self):
        """ Tests `arccos` function. """
        assert oeop.arccos(1) == 0
        xr.testing.assert_equal(
            oeop.arccos(self.test_data.xr_data_factor(-1, 1)), self.test_data.xr_data_factor(oeop.pi(), 0))

    def test_cosh(self):
        """ Tests `cosh` function. """
        assert oeop.cosh(0) == 1
        xr.testing.assert_equal(
            oeop.cosh(self.test_data.xr_data_factor(0, 0)), self.test_data.xr_data_factor(1, 1))

    def test_arcosh(self):
        """ Tests `arcosh` function. """
        assert oeop.arcosh(1) == 0
        xr.testing.assert_equal(
            oeop.arcosh(self.test_data.xr_data_factor(1, 1)), self.test_data.xr_data_factor(0, 0))

    def test_sin(self):
        """ Tests `sin` function. """
        assert oeop.sin(0) == 0
        xr.testing.assert_equal(
            oeop.sin(self.test_data.xr_data_factor(0, oeop.pi()/2)), self.test_data.xr_data_factor(0, 1))

    def test_arcsin(self):
        """ Tests `arcsin` function. """
        assert oeop.arcsin(0) == 0
        xr.testing.assert_equal(
            oeop.arcsin(self.test_data.xr_data_factor(0, 1)), self.test_data.xr_data_factor(0, oeop.pi()/2))

    def test_sinh(self):
        """ Tests `sinh` function. """
        assert oeop.sinh(0) == 0
        xr.testing.assert_equal(
            oeop.sinh(self.test_data.xr_data_factor(0, 0)), self.test_data.xr_data_factor(0, 0))

    def test_arsinh(self):
        """ Tests `arsinh` function. """
        assert oeop.arsinh(0) == 0
        xr.testing.assert_equal(
            oeop.arsinh(self.test_data.xr_data_factor(0, 0)), self.test_data.xr_data_factor(0, 0))

    def test_tan(self):
        """ Tests `tan` function. """
        assert oeop.tan(0) == 0
        xr.testing.assert_equal(
            oeop.tan(self.test_data.xr_data_factor(0, 0)), self.test_data.xr_data_factor(0, 0))

    def test_arctan(self):
        """ Tests `arctan` function. """
        assert oeop.arctan(0) == 0
        xr.testing.assert_equal(
            oeop.arctan(self.test_data.xr_data_factor(0, 0)), self.test_data.xr_data_factor(0, 0))

    def test_tanh(self):
        """ Tests `tanh` function. """
        assert oeop.tanh(0) == 0
        xr.testing.assert_equal(
            oeop.tanh(self.test_data.xr_data_factor(0, 0)), self.test_data.xr_data_factor(0, 0))

    def test_artanh(self):
        """ Tests `artanh` function. """
        assert oeop.artanh(0) == 0
        xr.testing.assert_equal(
            oeop.artanh(self.test_data.xr_data_factor(0, 0)), self.test_data.xr_data_factor(0, 0))

    def test_arctan2(self):
        """ Tests `arctan2` function. """
        assert oeop.arctan2(0, 0) == 0
        assert np.isnan(oeop.arctan2(np.nan, 1.5))
        xr.testing.assert_equal(
            oeop.arctan2(self.test_data.xr_data_factor(0, np.nan), self.test_data.xr_data_factor(0, 1.5)), self.test_data.xr_data_factor(0, np.nan))

    def test_linear_scale_range(self):
        """ Tests `linear_scale_range` function. """
        assert oeop.linear_scale_range(0.3, inputMin=-1, inputMax=1, outputMin=0, outputMax=255) == 165.75
        assert oeop.linear_scale_range(25.5, inputMin=0, inputMax=255) == 0.1
        assert np.isnan(oeop.linear_scale_range(np.nan, inputMin=0, inputMax=100))
        xr.testing.assert_equal(
            oeop.linear_scale_range(self.test_data.xr_data_factor(25.5, 51), inputMin=0, inputMax=255), self.test_data.xr_data_factor(0.1, 0.2))

    def test_scale(self):
        """ Tests `scale` function. """
        arr = np.random.randn(10)
        assert np.all(oeop.scale(arr) == arr)
        xr.testing.assert_equal(
            oeop.scale(self.test_data.xr_data_factor(1, 2), 8),
            self.test_data.xr_data_factor(8, 16))

    def test_mod(self):
        """ Tests `mod` function. """
        assert oeop.mod(27, 5) == 2
        assert oeop.mod(-27, 5) == 3
        assert oeop.mod(27, -5) == -3
        assert oeop.mod(-27, -5) == -2
        assert oeop.mod(27, 5) == 2
        assert np.isnan(oeop.mod(27, np.nan))
        assert np.isnan(oeop.mod(np.nan, 5))
        xr.testing.assert_equal(
            oeop.mod(self.test_data.xr_data_factor(27, -27), self.test_data.xr_data_factor(5, 5)),
            self.test_data.xr_data_factor(2, 3))

    def test_absolute(self):
        """ Tests `absolute` function. """
        assert oeop.absolute(0) == 0
        assert oeop.absolute(3.5) == 3.5
        assert oeop.absolute(-0.4) == 0.4
        assert oeop.absolute(-3.5) == 3.5
        xr.testing.assert_equal(
            oeop.absolute(self.test_data.xr_data_factor(0, -3.5)),
            self.test_data.xr_data_factor(0, 3.5))

    def test_sgn(self):
        """ Tests `sgn` function. """
        assert oeop.sgn(-2) == -1
        assert oeop.sgn(3.5) == 1
        assert oeop.sgn(0) == 0
        assert np.isnan(oeop.sgn(np.nan))
        xr.testing.assert_equal(
            oeop.sgn(self.test_data.xr_data_factor(-2, 3.5)),
            self.test_data.xr_data_factor(-1, 1))

    def test_sqrt(self):
        """ Tests `sqrt` function. """
        assert oeop.sqrt(0) == 0
        assert oeop.sqrt(1) == 1
        assert oeop.sqrt(9) == 3
        assert np.isnan(oeop.sqrt(np.nan))
        xr.testing.assert_equal(
            oeop.sqrt(self.test_data.xr_data_factor(9, 4)),
            self.test_data.xr_data_factor(3, 2))
        xr.testing.assert_equal(
            oeop.sqrt(self.test_data.xr_data_factor(np.nan, -4)),
            self.test_data.xr_data_factor(np.nan, np.nan))

    def test_power(self):
        """ Tests `power` function. """
        assert oeop.power(0, 2) == 0
        assert oeop.power(2.5, 0) == 1
        assert oeop.power(3, 3) == 27
        assert oeop.round(oeop.power(5, -1), 1) == 0.2
        assert oeop.power(1, 0.5) == 1
        assert oeop.power(1, None) is None
        assert oeop.power(None, 2) is None
        xr.testing.assert_equal(
            oeop.power(self.test_data.xr_data_factor(2, 3), 3),
            self.test_data.xr_data_factor(8, 27))

    def test_mean(self):
        """ Tests `mean` function. """
        assert oeop.mean([1, 0, 3, 2]) == 1.5
        assert oeop.mean([9, 2.5, np.nan, -2.5]) == 3
        assert np.isnan(oeop.mean([1, np.nan], ignore_nodata=False))
        assert np.isnan(oeop.mean([]))
        xr.testing.assert_equal(
            oeop.mean(self.test_data.xr_data_factor(3, 5)),
            xr.DataArray(4))

    def test_min(self):
        """ Tests `min` function. """
        assert oeop.min([1, 0, 3, 2]) == 0
        assert oeop.min([5, 2.5, np.nan, -0.7]) == -0.7
        assert np.isnan(oeop.min([1, 0, 3, np.nan, 2], ignore_nodata=False))
        assert np.isnan(oeop.min([np.nan, np.nan]))
        assert (oeop.min(self.test_data.xr_data_factor(3, 5), dimension = 'time') == 3).all()
        assert (oeop.min(self.test_data.xr_data_factor(np.nan, 5), dimension='time') == 5).all()

    def test_max(self):
        """ Tests `max` function. """
        assert oeop.max([1, 0, 3, 2]) == 3
        assert oeop.max([5, 2.5, np.nan, -0.7]) == 5
        assert np.isnan(oeop.max([1, 0, 3, np.nan, 2], ignore_nodata=False))
        assert np.isnan(oeop.max([np.nan, np.nan]))
        assert (oeop.max(self.test_data.xr_data_factor(3, 5), dimension = 'time') == 5).all()
        assert (oeop.max(self.test_data.xr_data_factor(np.nan, 5), dimension='time') == (oeop.max(self.test_data.xr_data_factor(3, 5), dimension = 'time'))).all()

    def test_median(self):
        """ Tests `median` function. """
        assert oeop.median([1, 3, 3, 6, 7, 8, 9]) == 6
        assert oeop.median([1, 2, 3, 4, 5, 6, 8, 9]) == 4.5
        assert oeop.median([-1, -0.5, np.nan, 1]) == -0.5
        assert np.isnan(oeop.median([-1, 0, np.nan, 1], ignore_nodata=False))
        assert np.isnan(oeop.median([]))
        xr.testing.assert_equal(
            oeop.median(self.test_data.xr_data_factor(3, 5)),
            xr.DataArray(4))

    def test_sd(self):
        """ Tests `sd` function. """
        assert oeop.sd([-1, 1, 3, np.nan]) == 2
        assert np.isnan(oeop.sd([-1, 1, 3, np.nan], ignore_nodata=False))
        assert np.isnan(oeop.sd([]))
        xr.testing.assert_equal(
            oeop.sd(self.test_data.xr_data_factor(3, 5)),
            xr.DataArray(1))

    def test_variance(self):
        """ Tests `variance` function. """
        assert oeop.variance([-1, 1, 3]) == 4
        assert oeop.variance([2, 3, 3, np.nan, 4, 4, 5]) == 1.1
        assert np.isnan(oeop.variance([-1, 1, np.nan, 3], ignore_nodata=False))
        assert np.isnan(oeop.variance([]))
        xr.testing.assert_equal(
            oeop.variance(self.test_data.xr_data_factor(3, 5)),
            xr.DataArray(1))

    def test_extrema(self):
        """ Tests `extrema` function. """
        self.assertListEqual(oeop.extrema([1, 0, 3, 2]), [0, 3])
        self.assertListEqual(oeop.extrema([5, 2.5, np.nan, -0.7]), [-0.7, 5])
        assert np.isclose(oeop.extrema([1, 0, 3, np.nan, 2], ignore_nodata=False), [np.nan, np.nan],
                          equal_nan=True).all()
        assert np.isclose(oeop.extrema([]), [np.nan, np.nan], equal_nan=True).all()
        xr.testing.assert_equal(
            oeop.extrema(self.test_data.xr_data_factor(3, 5)),
            xr.DataArray(np.append(3,5)))

    def test_clip(self):
        """ Tests `clip` function. """
        assert oeop.clip(-5, min=-1, max=1) == -1
        assert oeop.clip(10.001, min=1, max=10) == 10
        assert oeop.clip(0.000001, min=0, max=0.02) == 0.000001
        assert oeop.clip(None, min=0, max=1) is None

        # test array clipping
        assert np.isclose(oeop.clip([-2, -1, 0, 1, 2], min=-1, max=1), [-1, -1, 0, 1, 1], equal_nan=True).all()
        assert np.isclose(oeop.clip([-0.1, -0.001, np.nan, 0, 0.25, 0.75, 1.001, np.nan], min=0, max=1),
                          [0, 0, np.nan, 0, 0.25, 0.75, 1, np.nan], equal_nan=True).all()
        xr.testing.assert_equal(
            oeop.clip(self.test_data.xr_data_factor(-5, 2), min = 1, max = 8),
            self.test_data.xr_data_factor(1, 2))
        xr.testing.assert_equal(
            oeop.clip(self.test_data.xr_data_factor(1, 9), min=1, max=8),
            self.test_data.xr_data_factor(1, 8))
        xr.testing.assert_equal(
            oeop.clip(self.test_data.xr_data_factor(np.nan, 9), min=1, max=8),
            self.test_data.xr_data_factor(np.nan, 8))

    def test_quantiles(self):
        """ Tests `quantiles` function. """
        quantiles_1 = oeop.quantiles(data=[2, 4, 4, 4, 5, 5, 7, 9], probabilities=[0.005, 0.01, 0.02, 0.05, 0.1, 0.5])
        quantiles_1 = [oeop.round(quantile, p=2) for quantile in quantiles_1]
        assert quantiles_1 == [2.07, 2.14, 2.28, 2.7, 3.4, 4.5]
        quantiles_2 = oeop.quantiles(data=[2, 4, 4, 4, 5, 5, 7, 9], q=4)
        quantiles_2 = [oeop.round(quantile, p=2) for quantile in quantiles_2]
        assert quantiles_2 == [4, 4.5, 5.5]
        quantiles_3 = oeop.quantiles(data=[-1, -0.5, np.nan, 1], q=2)
        quantiles_3 = [oeop.round(quantile, p=2) for quantile in quantiles_3]
        assert quantiles_3 == [-0.5]
        quantiles_4 = oeop.quantiles(data=[-1, -0.5, np.nan, 1], q=4, ignore_nodata=False)
        assert np.all([np.isnan(quantile) for quantile in quantiles_4]) and len(quantiles_4) == 3
        quantiles_5 = oeop.quantiles(data=[], probabilities=[0.1, 0.5])
        assert np.all([np.isnan(quantile) for quantile in quantiles_5]) and len(quantiles_5) == 2
        assert (oeop.quantiles(self.test_data.xr_data_factor(1, 2), dimension = None, q = 2) == xr.DataArray(np.array([1.5, 1.5, 1.5]))).all()
        assert (oeop.quantiles(self.test_data.xr_data_factor(1, 2), dimension='time', q=2) == xr.DataArray(
            np.array([1.5, 1.5, 1.5]))).all()
        assert (oeop.quantiles(self.test_data.xr_data_factor(np.nan, 2), dimension='time', q=2) == xr.DataArray(
            np.array([2, 2, 2]))).all()

    def test_cummin(self):
        """ Tests `cummin` function. """
        self.assertListEqual(oeop.cummin([5, 3, 1, 3, 5]).tolist(), [5, 3, 1, 1, 1])
        assert np.isclose(oeop.cummin([5, 3, np.nan, 1, 5]), [5, 3, np.nan, 1, 1], equal_nan=True).all()
        assert np.isclose(oeop.cummin([5, 3, np.nan, 1, 5], ignore_nodata=False),
                          [5, 3, np.nan, np.nan, np.nan], equal_nan=True).all()
        assert (oeop.cummin(xr.DataArray(np.array([3, 5, 2]))) == [3, 3, 2]).all()

    def test_cummax(self):
        """ Tests `cummax` function. """
        self.assertListEqual(oeop.cummax([1, 3, 5, 3, 1]).tolist(), [1, 3, 5, 5, 5])
        assert np.isclose(oeop.cummax([1, 3, np.nan, 5, 1]), [1, 3, np.nan, 5, 5], equal_nan=True).all()
        assert np.isclose(oeop.cummax([1, 3, np.nan, 5, 1], ignore_nodata=False),
                          [1, 3, np.nan, np.nan, np.nan], equal_nan=True).all()
        assert (oeop.cummax(xr.DataArray(np.array([3, 5, 2]))) == [3, 5, 5]).all()

    def test_cumproduct(self):
        """ Tests `cumproduct` function. """
        self.assertListEqual(oeop.cumproduct([1, 3, 5, 3, 1]).tolist(), [1, 3, 15, 45, 45])
        assert np.isclose(oeop.cumproduct([1, 2, 3, np.nan, 3, 1]), [1, 2, 6, np.nan, 18, 18], equal_nan=True).all()
        assert np.isclose(oeop.cumproduct([1, 2, 3, np.nan, 3, 1], ignore_nodata=False),
                          [1, 2, 6, np.nan, np.nan, np.nan], equal_nan=True).all()

    def test_cumsum(self):
        """ Tests `cumsum` function. """
        self.assertListEqual(oeop.cumsum([1, 3, 5, 3, 1]).tolist(), [1, 4, 9, 12, 13])
        assert np.isclose(oeop.cumsum([1, 3, np.nan, 3, 1]), [1, 4, np.nan, 7, 8], equal_nan=True).all()
        assert np.isclose(oeop.cumsum([1, 3, np.nan, 3, 1], ignore_nodata=False),
                          [1, 4, np.nan, np.nan, np.nan], equal_nan=True).all()

    def test_sum(self):
        """ Tests `sum` function. """
        assert oeop.sum([5, 1]) == 6
        assert oeop.sum([-2, 4, 2.5]) == 4.5
        assert np.isnan(oeop.sum([1, np.nan], ignore_nodata=False))

        # xarray tests
        # Take sum over 't' dimension in a 3d array
        self.assertEqual(
            int(oeop.sum(self.test_data.xr_data_3d)[0, 0].data),
            88
            )
        # Take sum over 't' dimension in a 3d array
        self.assertEqual(
            list(oeop.sum([self.test_data.xr_data_3d, 1000])[:, 0, 0].data),
            [1008., 1080.]
            )
        # Take sum over 's' dimension in a 4d array
        self.assertListEqual(
            list(oeop.sum(self.test_data.xr_data_4d)[:, 0, 0].data),
            [14, 140]
            )
        # Test with input as [xr.DataArray, xr.DataArray]
        self.assertEqual(
            (
             oeop.sum([self.test_data.xr_data_3d, self.test_data.xr_data_3d]) -
             self.test_data.xr_data_3d * 2
            ).sum(), 0)

    def test_product(self):
        """ Tests `product` function. """
        assert oeop.product([5, 0]) == 0
        assert oeop.product([-2, 4, 2.5]) == -20
        assert np.isnan(oeop.product([1, np.nan], ignore_nodata=False))
        assert oeop.product([-1]) == -1
        assert np.isnan(oeop.product([np.nan], ignore_nodata=False))
        assert np.isnan(oeop.product([]))

        C = np.ones((2, 5, 5)) * 100
        assert np.sum(oeop.product(C) - np.ones((5, 5)) * 10000) == 0
        assert np.sum(oeop.product(deepcopy(C), extra_values=[2]) - np.ones((5, 5)) * 20000) == 0
        assert np.sum(oeop.product(deepcopy(C), extra_values=[2, 3]) - np.ones((5, 5)) * 60000) == 0

        # xarray tests
        # Take sum over 't' dimension in a 3d array
        self.assertEqual(
            int(oeop.product(self.test_data.xr_data_3d)[0, 0].data),
            640
            )
        # Take sum over 's' dimension in a 4d array
        self.assertListEqual(
            list(oeop.product(self.test_data.xr_data_4d)[:, 0, 0].data),
            [64, 64000]
            )

    def test_add(self):
        """ Tests `add` function. """
        assert oeop.add(5, 2.5) == 7.5
        assert oeop.add(-2, -4) == -6
        assert oeop.add(1, None) is None
        xr.testing.assert_equal(
            oeop.add(self.test_data.xr_data_factor(1, 9), self.test_data.xr_data_factor(2, 2)),
            self.test_data.xr_data_factor(3, 11))
        xr.testing.assert_equal(
            oeop.add(self.test_data.xr_data_factor(1, 9), self.test_data.xr_data_factor(np.nan, -2)),
            self.test_data.xr_data_factor(np.nan, 7))

    def test_subtract(self):
        """ Tests `subtract` function. """
        assert oeop.subtract(5, 2.5) == 2.5
        assert oeop.subtract(-2, 4) == -6
        assert oeop.subtract(1, None) is None

        # xarray tests
        assert (oeop.subtract(self.test_data.xr_data_3d,
                             self.test_data.xr_data_3d)).sum() == 0
        
    def test_multiply(self):
        """ Tests `multiply` function. """
        assert oeop.multiply(5, 2.5) == 12.5
        assert oeop.multiply(-2, -4) == 8
        assert oeop.multiply(1, None) is None
        xr.testing.assert_equal(
            oeop.multiply(self.test_data.xr_data_factor(3, 9), self.test_data.xr_data_factor(2, np.nan)),
            self.test_data.xr_data_factor(6, np.nan))

    def test_divide(self):
        """ Tests `divide` function. """
        assert oeop.divide(5, 2.5) == 2.
        assert oeop.divide(-2, 4) == -0.5
        assert oeop.divide(1, None) is None
        xr.testing.assert_equal(
            oeop.divide(self.test_data.xr_data_factor(1, 6.4), self.test_data.xr_data_factor(np.nan, 2)),
            self.test_data.xr_data_factor(np.nan, 3.2))

    def test_normalized_difference(self):
        """ Tests `normalized_difference` function. """
        assert oeop.normalized_difference(5, 3) == 0.25
        assert oeop.normalized_difference(1, 1) == 0
        assert (oeop.normalized_difference(np.array([1, 1]), np.array([0, 1])) == np.array([1, 0])).all()
        xr.testing.assert_equal(oeop.normalized_difference(self.test_data.xr_data_factor(1, 5), self.test_data.xr_data_factor(1, 3)), self.test_data.xr_data_factor(0, 0.25))
        
    def test_apply_kernel(self):
        """ Tests `apply_kernel` function. """
        # xarray tests
        kernel = np.asarray([[0,0,0],[0,1,0],[0,0,0]])
        # With the given kernel the result must be the same as the input
        xr.testing.assert_equal(oeop.apply_kernel(self.test_data.xr_data_4d,kernel,border=0, factor=1),self.test_data.xr_data_4d)
        xr.testing.assert_equal(oeop.apply_kernel(self.test_data.xr_data_3d,kernel,border=0, factor=1),self.test_data.xr_data_3d)

if __name__ == "__main__":
    unittest.main()
