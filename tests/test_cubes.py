"""
Most tests are in alignment with:
https://openeo.org/documentation/1.0/processes.html
"""

import os
import unittest
import pytest
import openeo_processes as oeop
import xarray as xr
import numpy as np
import pandas as pd


@pytest.mark.usefixtures("test_data")
class CubesTester(unittest.TestCase):
    """ Tests all cubes functions. """

    def test_reduce_dimension(self):
        """ Tests `reduce_dimension` function. """

        # xarray tests
        # Reduce spectral dimension using the process `sum`
        # Take sum over 's' dimension in a 4d array
        reduced = oeop.reduce_dimension(self.test_data.xr_data_4d, reducer=oeop.sum, dimension='bands')
        self.assertListEqual(
            list(reduced[:, 0, 0].data),
            [14, 140]
            )

    def test_merge_cubes(self):
        """Tests 'merge_cubes' function. """
        merged = oeop.merge_cubes(self.test_data.xr_data_4d, self.test_data.xr_data_4d,
                                  oeop.add)  # merges two cubes together with add: x + x
        assert (merged.dims == self.test_data.xr_data_4d.dims)  # dimensions did not change
        xr.testing.assert_equal(merged, self.test_data.xr_data_4d * 2)  # x + x is the same as the cube*2
        xr.testing.assert_equal(
            oeop.merge_cubes(self.test_data.xr_data_factor(5, 9), self.test_data.xr_data_factor(2, 3), oeop.subtract),
            self.test_data.xr_data_factor(3, 6))
        merged2 = oeop.merge_cubes(self.test_data.xr_data_factor(5, 9)[:, :3],
                                   self.test_data.xr_data_factor(2, 7)[:, 3:])
        assert (merged2.dims == self.test_data.xr_data_factor(5, 7).dims)
        xr.testing.assert_equal(
            oeop.merge_cubes(self.test_data.xr_data_factor(5, 9).isel(time=0),
                             self.test_data.xr_data_factor(2, 3).isel(time=1)),
            self.test_data.xr_data_factor(5, 3))
        xr.testing.assert_equal(
            oeop.merge_cubes(self.test_data.xr_data_factor(5, 9).isel(time=0),
                             self.test_data.xr_data_factor(2, 3).isel(time=0), oeop.add),
            self.test_data.xr_data_factor(7, 3).isel(time=0))
        merged3 = oeop.merge_cubes(self.test_data.xr_data_factor(5, 9), self.test_data.xr_data_factor(2, 3))
        assert (merged3.shape == (2, 2, 5, 3))  # added first dimension, so shape is now longer

    def test_save_result(self):
        """ Tests `reduce_dimension` function. """

        # TODO improve file check
        # xarray tests
        out_filename = "out.tif"
        reduced = oeop.reduce_dimension(self.test_data.xr_data_3d, reducer=oeop.min, dimension='time')
        oeop.save_result(reduced, out_filename)
        assert os.path.exists(out_filename)
        os.remove(out_filename)
        
        reduced = oeop.reduce_dimension(self.test_data.xr_data_4d, reducer=oeop.min, dimension='time')
        print('reduced:',reduced)
        oeop.save_result(reduced, out_filename)
        assert os.path.exists(out_filename)
        os.remove(out_filename)
        
        out_filename = "out.nc"
        oeop.save_result(self.test_data.xr_data_3d, out_filename, format='netCDF')
        assert os.path.exists(out_filename)
        os.remove(out_filename)

        oeop.save_result(self.test_data.xr_data_3d, format='netCDF')
        assert os.path.exists('out.nc')
        os.remove('out.nc')
        
        oeop.save_result(self.test_data.xr_data_4d, out_filename, format='netCDF')
        assert os.path.exists('out.nc')
        os.remove('out.nc')
        
        oeop.save_result(self.test_data.xr_data_4d, format='netCDF')
        assert os.path.exists('out.nc')
        os.remove('out.nc')

    def test_fit_curve(self):
        """Tests 'fit_curve' function. """
        rang = np.linspace(0, 4 * np.pi, 24)
        rang = [np.cos(rang) + 0.5 * np.sin(rang) + np.random.rand(24) * 0.1,
                np.cos(rang) + 0.5 * np.sin(rang) + np.random.rand(
                    24) * 0.2]  # define data with y = 0 + 1 * cos() + 0.5 *sin()
        xdata = xr.DataArray(rang, coords=[["NY", "LA"], pd.date_range("2000-01-01", periods=24, freq='M')],
                             dims=["space", "time"])

        def func(x, a, b, c):
            return a + b * np.cos(2 * np.pi / 31557600 * x) + c * np.sin(2 * np.pi / 31557600 * x)

        params = (oeop.fit_curve(xdata, parameters=(0, 1), function=func, dimension='time'))
        assert (np.isclose(params, [0, 1, 0.5], atol=0.3)).all()  # output should be close to 0, 1, 0.5

    def test_predict_curve(self):
        """Tests 'predict_curve' function. """
        rang = np.linspace(0, 4 * np.pi, 24)
        rang = [np.cos(rang) + 0.5 * np.sin(rang) + np.random.rand(24) * 0.1,
                np.cos(rang) + 0.5 * np.sin(rang) + np.random.rand(
                    24) * 0.2]  # define data with y = 0 + 1 * cos() + 0.5 *sin()
        xdata = xr.DataArray(rang, coords=[["NY", "LA"], pd.date_range("2000-01-01", periods=24, freq='M')],
                             dims=["space", "time"])

        def func(x, a, b, c):
            return a + b * np.cos(2 * np.pi / 31557600 * x) + c * np.sin(2 * np.pi / 31557600 * x)

        params = (oeop.fit_curve(xdata, parameters=(0, 1), function=func, dimension='time'))
        predicted = oeop.predict_curve(xdata, params, func, dimension='time',
                                       labels=pd.date_range("2002-01-01", periods=24, freq='M'))
        assert xdata.dims == predicted.dims
        assert (predicted < 1.5).all()
        predicted = oeop.predict_curve(xdata, params, func, dimension='time',
                                       labels=pd.date_range("2000-01-01", periods=24, freq='M'))
        assert (np.isclose(xdata, predicted, atol=0.3)).all()


if __name__ == "__main__":
    unittest.main()
