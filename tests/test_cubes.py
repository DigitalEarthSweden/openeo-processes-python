"""
Most tests are in alignment with:
https://openeo.org/documentation/1.0/processes.html
"""

import os
import unittest
import pytest
import openeo_processes as oeop
import xarray as xr


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


if __name__ == "__main__":
    unittest.main()
