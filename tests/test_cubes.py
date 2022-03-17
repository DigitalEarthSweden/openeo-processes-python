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
import geopandas as gpd


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
        # TODO improve file check
        # xarray tests
        out_filename = "out.tif"
        out_filename_0 = "out_00000.tif"
        out_filename_1 = "out_00001.tif"
        out_product = "product.yml"
        oeop.save_result(self.test_data.xr_odc_data_3d, out_filename)
        assert os.path.exists(out_filename_0)
        assert os.path.exists(out_product)
        os.remove(out_filename_0)
        os.remove(out_product)

        oeop.save_result(self.test_data.xr_odc_data_4d, out_filename)
        assert os.path.exists(out_filename_0)
        assert os.path.exists(out_filename_1)
        assert os.path.exists(out_product)
        os.remove(out_filename_0)
        os.remove(out_filename_1)
        os.remove(out_product)

        out_filename = "out.nc"
        out_filename_0 = "out_00000.nc"
        out_filename_1 = "out_00001.nc"
        out_filename_combined = "out_combined.nc"

        oeop.save_result(self.test_data.xr_odc_data_3d, out_filename, format='netCDF')
        assert os.path.exists(out_filename_0)
        assert os.path.exists(out_product)
        os.remove(out_filename_0)
        os.remove(out_product)

        oeop.save_result(self.test_data.xr_odc_data_3d, format='netCDF')
        assert os.path.exists(out_filename_0)
        assert os.path.exists(out_product)
        os.remove(out_filename_0)
        os.remove(out_product)

        oeop.save_result(self.test_data.xr_odc_data_4d, format='netCDF')
        assert os.path.exists(out_filename_0)
        assert os.path.exists(out_filename_1)
        assert os.path.exists(out_filename_combined)
        assert os.path.exists(out_product)
        os.remove(out_filename_0)
        os.remove(out_filename_1)
        os.remove(out_filename_combined)
        os.remove(out_product)

    def test_save_result_from_file(self):
        src = os.path.join(os.path.dirname(__file__), "data", "out.time.nc")
        ref_ds = xr.load_dataset(src)
        ref_ds_0 = ref_ds.loc[dict(time="2016-01-13T12:00:00.000000000")]
        data_array = ref_ds.to_array(dim="bands")
        oeop.save_result(data_array, format='netCDF')
        actual_ds_0 = xr.load_dataset("out_00000.nc")
        assert ref_ds_0.dims == actual_ds_0.dims
        assert ref_ds_0.coords == actual_ds_0.coords
        assert ref_ds_0.variables == actual_ds_0.variables
        assert ref_ds_0.geobox == actual_ds_0.geobox
        assert ref_ds_0.extent == actual_ds_0.extent
        assert "crs" in actual_ds_0.attrs and actual_ds_0.attrs["crs"] == 'PROJCRS["Azimuthal_Equidistant",BASEGEOGCRS["WGS 84",DATUM["World Geodetic System 1984",ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]],ID["EPSG",6326]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]]],CONVERSION["unnamed",METHOD["Modified Azimuthal Equidistant",ID["EPSG",9832]],PARAMETER["Latitude of natural origin",53,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8801]],PARAMETER["Longitude of natural origin",24,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8802]],PARAMETER["False easting",5837287.81977,LENGTHUNIT["metre",1],ID["EPSG",8806]],PARAMETER["False northing",2121415.69617,LENGTHUNIT["metre",1],ID["EPSG",8807]]],CS[Cartesian,2],AXIS["easting",east,ORDER[1],LENGTHUNIT["metre",1,ID["EPSG",9001]]],AXIS["northing",north,ORDER[2],LENGTHUNIT["metre",1,ID["EPSG",9001]]]]'
        assert "datetime_from_dim" in actual_ds_0.attrs
        assert actual_ds_0.result.dims == ("y", "x")
        for i in range(10):
            os.remove(f"out_{str(i).zfill(5)}.nc")
        os.remove("out_combined.nc")
        os.remove("product.yml")

    def test_fit_curve(self):
        """Tests 'fit_curve' function. """
        rang = np.linspace(0, 4 * np.pi, 24)
        rang = [np.cos(rang) + 0.5 * np.sin(rang) + np.random.rand(24) * 0.1,
                np.cos(rang) + 0.5 * np.sin(rang) + np.random.rand(
                    24) * 0.2]  # define data with y = 0 + 1 * cos() + 0.5 *sin()
        xdata = xr.DataArray(rang, coords=[["NY", "LA"], pd.date_range("2000-01-01", periods=24, freq='M')],
                             dims=["space", "time"])

        def func_oeop(x, *parameters):
            _2sjyaa699_11 = oeop.pi(**{})
            _9k6vt7qcn_2 = oeop.multiply(**{'x': 2, 'y': _2sjyaa699_11})
            _p42lrxmbq_16 = oeop.divide(**{'x': _9k6vt7qcn_2, 'y': 31557600})
            _wz26aglyi_5 = oeop.multiply(**{'x': _p42lrxmbq_16, 'y': x})
            _v81bsalku_7 = oeop.cos(**{'x': _wz26aglyi_5})
            _32frj455b_1 = oeop.pi(**{})
            _lyjcuq5vd_15 = oeop.multiply(**{'x': 2, 'y': _32frj455b_1})
            _1ipvki94n_4 = oeop.divide(**{'x': _lyjcuq5vd_15, 'y': 31557600})
            _ya3hbxpot_17 = oeop.multiply(**{'x': _1ipvki94n_4, 'y': x})
            _0p7xlqeyo_8 = oeop.sin(**{'x': _ya3hbxpot_17})

            _kryhimf6r_6 = oeop.array_element(**{'data': parameters, 'index': 0})
            _jxs4umqsh_10 = oeop.array_element(**{'data': parameters, 'index': 1})
            _8jjjztmya_12 = oeop.array_element(**{'data': parameters, 'index': 2})

            _jhus2gz74_13 = oeop.multiply(**{'x': _jxs4umqsh_10, 'y': _v81bsalku_7})
            _0v09jn699_14 = oeop.multiply(**{'x': _8jjjztmya_12, 'y': _0p7xlqeyo_8})
            _xb4c1hk1f_9 = oeop.add(**{'x': _kryhimf6r_6, 'y': _jhus2gz74_13})
            _b4mf181yp_3 = oeop.add(**{'x': _xb4c1hk1f_9, 'y': _0v09jn699_14})
            return _b4mf181yp_3

        params = (oeop.fit_curve(xdata, parameters=[1, 1, 1], function=func_oeop, dimension='time'))
        assert (np.isclose(params, [0, 1, 0.5], atol=0.3)).all()  # output should be close to 0, 1, 0.5
        params_2 = (oeop.fit_curve(xdata, parameters=params, function=func_oeop, dimension='time'))
        assert (np.isclose(params_2, [0, 1, 0.5], atol=0.3)).all()
        assert (np.isclose(params, params_2, atol=0.01)).all()

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

        params = (oeop.fit_curve(xdata, parameters=(0, 1, 0), function=func, dimension='time'))
        predicted = oeop.predict_curve(xdata, params, func, dimension='time',
                                       labels=pd.date_range("2002-01-01", periods=24, freq='M'))
        assert xdata.dims == predicted.dims
        assert (predicted < 1.8).all()
        predicted = oeop.predict_curve(xdata, params, func, dimension='time',
                                       labels=pd.date_range("2000-01-01", periods=24, freq='M'))
        assert (np.isclose(xdata, predicted, atol=0.5)).all()
        dim_times = oeop.dimension_labels(self.test_data.xr_data_factor(), 't')
        predicted_dim_labels = (oeop.predict_curve(xdata, params, func, dimension='time', labels=dim_times))
        assert xdata.dims == predicted_dim_labels.dims
        assert (predicted_dim_labels < 1.8).all()
        xdata_t = xr.DataArray(rang, coords=[["NY", "LA"], pd.date_range("2000-01-01", periods=24, freq='M')],
                               dims=["space", "t"])
        predicted_t = oeop.predict_curve(xdata_t, params, func, dimension='t',
                                         labels=pd.date_range("2000-01-01", periods=24, freq='M'))
        xr.testing.assert_equal(predicted, predicted_t)
        predicted_time = oeop.predict_curve(xdata, params, func, dimension='time',
                                       labels=pd.date_range("2002-01-01", periods=2, freq='M'))
        predicted_str = oeop.predict_curve(xdata, params, func, dimension='time',
                                       labels=["2002-01-31 00:00", "2002-02-28"])
        assert (predicted_time.values == predicted_str.values).all()


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

    def test_create_raster_cube(self):
        """Tests 'create_raster_cube' function. """
        assert len(oeop.create_raster_cube()) == 0

    def test_add_dimension(self):
        """Tests 'add_dimension' function. """
        assert oeop.add_dimension(self.test_data.xr_data_factor(), 'cubes', 'Cube01').shape == (1, 2, 5, 3)

    def test_dimension_labels(self):
        """Tests 'dimension_labels' function. """
        assert (oeop.dimension_labels(self.test_data.xr_data_factor(), 'x') == [118.9, 119.9, 120.9]).all()
        assert (oeop.dimension_labels(self.test_data.xr_data_factor(), 't') == oeop.dimension_labels(self.test_data.xr_data_factor(), 'time')).all()

    def test_drop_dimension(self):
        """Tests 'drop_dimension' function. """
        data = oeop.add_dimension(self.test_data.xr_data_factor(), 'cubes', 'Cube01')
        xr.testing.assert_equal(oeop.drop_dimension(data, 'cubes'), self.test_data.xr_data_factor())

    def test_rename_dimension(self):
        """Tests 'rename_dimension' function. """
        data = oeop.rename_dimension(self.test_data.xr_data_factor(), 'x', 'longitude')
        assert (data.dims == ('time', 'y', 'longitude'))

    def test_rename_labels(self):
        """Tests 'rename_labels' function. """
        data = oeop.rename_labels(self.test_data.xr_data_factor(), 'x', [119, 120, 121], [118.9, 119.9, 120.9])
        assert (data['x'].values == (119, 120, 121)).all()

    def test_filter_temporal(self):
        """Tests 'filter_temporal' function. """
        data = oeop.filter_temporal(self.test_data.xr_data_factor(), ['2019-12-01', '2019-12-05'])
        data2 = oeop.filter_temporal(self.test_data.xr_data_factor(), ['2019-12-01T00:00:00Z', '2019-12-02T00:00:00Z'])
        xr.testing.assert_equal(data, data2)

    def test_filter_spatial(self):
        """Tests 'filter_spatial' function. """
        geo = {'type': 'Polygon', 'coordinates': [[117.9, 55.2], [120.5, 58.4], [120.5, 55.2]]}
        assert (oeop.filter_spatial(self.test_data.xr_data_factor(), geo).dims == self.test_data.xr_data_factor(1, 1)[:,:4, :2].dims)

    def test_filter_labels(self):
        """Tests 'filter_labels' function. """
        xr.testing.assert_equal(oeop.filter_labels(self.test_data.xr_data_factor(), oeop.gt, 'x', {'y': 120}), self.test_data.xr_data_factor().loc[{'x': [120.9]}])

    def test_filter_bbox(self):
        """Tests 'filter_bbox' function. """
        extent = {'west': 60, 'east': 56, 'north': 120, 'south': 118, 'crs':'EPSG:4326'}
        xr.testing.assert_equal(oeop.filter_bbox(self.test_data.xr_data_factor(), extent), self.test_data.xr_data_factor().loc[{'x': [118.9, 119.9], 'y': [56.3, 57.3, 58.3, 59.3]}])
        extent = {'west': 63, 'east': 62, 'north': 124, 'south': 123, 'crs': 4326}
        assert len((oeop.filter_bbox(self.test_data.xr_data_factor(), extent)).values[0])==0

    def test_mask(self):
        """ Tests `mask` function. """
        assert (oeop.mask(np.array([[1,3,6],[2,2,2]]), np.array([[True,False,True],[False,False,True]]), 999) == np.array([[999,3,999],[2,2,999]])).all()
        xr.testing.assert_equal(oeop.mask(self.test_data.xr_data_factor(1, 5),self.test_data.xr_data_factor(True, False), replacement = 999),
                                self.test_data.xr_data_factor(999, 5))

    def test_mask_polygon(self):
        """Tests 'mask_polygon function. """
        geojson = {'type' : 'Polygon', 'coordinates': [[117.9, 55.2], [120.5, 58.4], [120.5, 55.2]]}
        assert ((oeop.mask_polygon(self.test_data.xr_data_factor(1,1), geojson))['x'].values == self.test_data.xr_data_factor(1,1)['x'].values).all()
        geojson = {'type': 'MultiPolygon', 'coordinates': [[(117.9, 55.2), (120.5, 58.4), (120.5, 55.2)], [(120.5, 55.2), (121, 58.4), (120.5, 55.2)]]}
        #print(oeop.mask_polygon(self.test_data.xr_data_factor(1,1), geojson))

    def test_aggregate_temporal_period(self):
        """ Tests 'aggregate_temporal_period' function. """
        assert (oeop.aggregate_temporal_period(self.test_data.xr_data_4d,'day',oeop.min) == self.test_data.xr_data_4d.values).all()
        xr.testing.assert_equal(oeop.aggregate_temporal_period(self.test_data.xr_data_4d,'day',oeop.min),
                                oeop.aggregate_temporal_period(self.test_data.xr_data_4d, 'day',oeop.max, 'time', {}))

    def test_apply_dimension(self):
        """Tests 'apply_dimension' function. """
        assert (oeop.apply_dimension(self.test_data.xr_data_factor(1,2), oeop.min, 'time', 'time').values ==
                                 self.test_data.xr_data_factor(1,2)[0].values).all()
        assert (oeop.apply_dimension(self.test_data.xr_data_factor(1, 2), oeop.min, 'time', 'time').dims ==
                self.test_data.xr_data_factor(1, 2).dims)


    def test_aggregate_spatial(self):
        """Tests 'aggregate_spatial' function. """
        vector_points = oeop.vector_to_regular_points(self.test_data.geojson_polygon, 0.01)
        assert type(oeop.aggregate_spatial(self.test_data.equi7xarray, vector_points, oeop.mean, 'result')) == gpd.geodataframe.GeoDataFrame

if __name__ == "__main__":
    unittest.main()
