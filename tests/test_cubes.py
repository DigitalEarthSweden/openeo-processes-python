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



    def test_resample_cube_temporal(self):
        """ Tests `reduce_dimension` function. """
        xdata = xr.DataArray(np.array([[1, 3], [7, 8]]),
                             coords=[["NY", "LA"], pd.date_range("2000-01-01", "2000-02-01", periods=2)],
                             dims=["space", "time"])
        target = xr.DataArray(np.array([[1, 3], [7, 8]]),
                              coords=[["NY", "LA"], pd.date_range("2000-01-10", "2000-02-10", periods=2)],
                              dims=["space", "time"])
        resample = oeop.resample_cube_temporal(xdata, target, dimension='time')
        xr.testing.assert_equal(resample, target)
        xdata2 = xr.DataArray(np.array([[1, 3, 4], [7, 8, 10]]),
                             coords=[["NY", "LA"], pd.date_range("2000-01-01", "2000-03-01", periods=3)],
                             dims=["space", "time"])
        resample2 = oeop.resample_cube_temporal(xdata2, target, dimension='time', valid_within=15)
        xr.testing.assert_equal(resample2, target)

if __name__ == "__main__":
    unittest.main()
