from datetime import datetime
from os.path import splitext
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import odc.algo
import rioxarray  # needed by save_result even if not directly called
import xarray as xr
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from openeo_processes.extension.odc import write_odc_product
from openeo_processes.utils import process, get_time_dimension_from_data, xarray_dataset_from_dask_dataframe, get_equi7_tiles, derive_datasets_and_filenames_from_tiles
from openeo_processes.errors import DimensionNotAvailable, TooManyDimensions
from scipy import optimize
from datetime import datetime
import datacube
import dask
from datacube.utils.cog import write_cog
try:
    from pyproj import Transformer, CRS
except ImportError:
    Transformer = None
    CRS = None
import xgboost as xgb
import dask.dataframe as df
from geocube.api.core import make_geocube
import dask_geopandas

import geopandas as gpd
import urllib, json
import os

from functools import partial

DEFAULT_CRS = 4326


###############################################################################
# Load Collection Process
###############################################################################


def odc_load_helper(odc_cube, params: Dict[str, Any]) -> xr.DataArray:
    """Helper method to load a xarray DataArray from ODC."""
    datacube = odc_cube.load(**params)

    # Improve CPU and MEM USAGE
    for name, data_var in datacube.data_vars.items():
        datacube[name] = datacube[name].where(datacube[name] != datacube[name].nodata)
    datacube.attrs['nodata'] = np.nan

    refdata = {'collection': params['product']}
    datacube.attrs.update(refdata)

    # Convert to xr.DataArray
    # TODO: add conversion with multiple and custom dimensions
    return datacube.to_array(dim='bands')


@process
def load_collection():
    """
    Returns class instance of `LoadCollection`.
    For more details, please have a look at the implementations inside
    `LoadCollection`.

    Returns
    -------
    LoadCollection :
        Class instance implementing all 'load_collection' processes.

    """
    return LoadCollection()


class LoadCollection:
    """
    Class implementing all 'load_collection' processes.

    """

    @staticmethod
    def exec_odc(odc_cube, product: str, dask_chunks: dict,
                 x: tuple = (), y: tuple = (), time: list = [],
                 measurements: list = [], crs: str = "EPSG:4326"):

        odc_params = {
            'product': product,
            'dask_chunks': dask_chunks
        }
        if x:
            odc_params['x'] = x
        if y:
            odc_params['y'] = y
        if crs:
            odc_params['crs'] = crs
        # lists are transformed to np.arrays by the wrapper
        # update when that step has been removed
        if len(time) > 0:
            odc_params['time'] = list(time)
        if len(measurements) > 0:
            odc_params['measurements'] = list(measurements)

        return odc_load_helper(odc_cube, odc_params)


###############################################################################
# Load Result Process
###############################################################################


@process
def load_result():
    """
    Returns class instance of `LoadResult`.
    For more details, please have a look at the implementations inside
    `LoadResult`.

    Returns
    -------
    LoadResult :
        Class instance implementing all 'load_result' processes.

    """
    return LoadResult()


class LoadResult:
    """
    Class implementing all 'load_result' processes.

    """

    @staticmethod
    def exec_odc(odc_cube, product: str, dask_chunks: dict,
                 x = None, y = None, time = [],
                 measurements = [], crs = None):

        odc_params = {
            'product': product,
            'dask_chunks': dask_chunks
        }
        part = False
        if crs:
            odc_params['crs'] = crs
        # lists are transformed to np.arrays by the wrapper
        # update when that step has been removed
        if len(time) > 0:
            odc_params['time'] = list(time)
        if len(measurements) > 0:
            odc_params['measurements'] = list(measurements)
        if x:
            odc_params['x'] = x
            part = True
        if y:
            odc_params['y'] = y
            part = True
        if part:
            try:
                dataarray = odc_load_helper(odc_cube, odc_params)
            except:
                if x:
                    odc_params['longitude'] = x
                    odc_params.pop('x')
                if y:
                    odc_params['latitude'] = y
                    odc_params.pop('y')
                dataarray = odc_load_helper(odc_cube, odc_params)
        else:
            dataarray = odc_load_helper(odc_cube, odc_params)

        # If data is in geographic coordinate system coords are called longitude/latitude
        # for consistency and easier handling in other processes rename them to x/y
        if "longitude" in dataarray.coords and "latitude" in dataarray.coords:
            dataarray.rename({"longitude": "x", "latitude": "y"})

        # In ODC each dataset must have a time dimension also if non exists
        # remove it if only a single one exist
        if "time" in dataarray.coords and len(dataarray["time"]) == 1:
            dataarray = dataarray.squeeze("time", drop=True)

        return dataarray


###############################################################################
# Save result process
###############################################################################


@process
def save_result():
    """
    Returns class instance of `save_result`.
    For more details, please have a look at the implementations inside
    `save_result`.

    Returns
    -------
    save_result :
        Class instance implementing all 'save_result' processes.

    """
    return SaveResult()


class SaveResult:
    """
    Class implementing 'save_result' processes.

    """

    @staticmethod
    def exec_xar(data, output_filepath='out', format='GTiff', options={}, write_prod: bool = True):
        """
        Save data to disk in specified format. Data is saved to files reflecting the EQUI7 Tiles that the original data
        is loaded from.

        Parameters
        ----------
        data : xr.DataArray
            An array of numbers. An empty array resolves always with np.nan.
        output_filepath: str
            Absolute or relative filepath where to store data on disk,
            with or without extention
        format: str, optional
            data format (default: GTiff)
        """
        formats = ['gtiff', 'netcdf', 'geotiff']
        if format.lower() in formats:
            format = format.lower()
        else:
            raise ValueError(f"Error when saving to file. Format '{format}' is not in {formats}.")

        data = data.fillna(-9999)
        data.attrs["nodata"] = -9999
        # Convert data array to data set, keeping a nice format on the bands.
        data = data.to_dataset(
            dim='bands'
        )

        if "crs" not in data.attrs:
            first_data_var = data.data_vars[list(data.data_vars.keys())[0]]
            data.attrs["crs"] = first_data_var.geobox.crs.to_wkt()

        tiles, gridder = get_equi7_tiles(data)
        
        # Renaming the time dimension
        if 'time' in data.dims:
            data = data.rename({'time': 't'})
        if 't' not in data.dims:
            data = data.assign_coords(t=datetime.now())
            data = data.expand_dims('t')

        # Add back the odc hack
        data.attrs["datetime_from_dim"] = str(datetime.now())

        # Avoid value error on attrs
        if hasattr(data, 't') and hasattr(data.t, 'units'):
            data.t.attrs.pop('units', None)

        if hasattr(data, 'grid_mapping'):
            data.attrs.pop('grid_mapping')

        # Group datasets by time. This will help with storing multiple files via dask.
        times, datasets = zip(*data.groupby("t"))

        if format == 'netcdf':
            ext = 'nc'
        else:
            ext = 'tif'

        final_datasets, dataset_filenames = derive_datasets_and_filenames_from_tiles(gridder, times, datasets, tiles, output_filepath, ext)
        if (len(final_datasets) == 0) or (len(dataset_filenames) == 0):
            raise Exception("No tiles could be derived from given dataset")

        # Submit list of netcdfs and filepaths to dask to compute
        if format == 'netcdf':
            xr.save_mfdataset(final_datasets, dataset_filenames)

        # Iterate datasets and save to tif
        elif format in ['gtiff','geotiff']:
            if len(final_datasets[0].dims) > 3:
                raise Exception("[!] Not possible to write a 4-dimensional GeoTiff, use NetCDF instead.")
            for idx, dataset in enumerate(final_datasets):
                dataset.rio.to_raster(raster_path=dataset_filenames[idx], **options)

        # Write and odc product yml file
        if write_prod:
            write_odc_product(datasets[0], output_filepath)


###############################################################################
# Reduce dimension process
###############################################################################


@process
def reduce_dimension():
    """
    Returns class instance of `reduce_dimension`.
    For more details, please have a look at the implementations inside
    `reduce_dimension`.

    Returns
    -------
    reduce_dimension :
        Class instance implementing all 'reduce_dimension' processes.

    """
    return ReduceDimension()


class ReduceDimension:
    """
    Class implementing all 'reduce_dimension' processes.

    """

    @staticmethod
    def exec_xar(data, reducer, dimension=None, context={}):
        """
        Parameters
        ----------
        data : xr.DataArray
            An array of numbers. An empty array resolves always with np.nan.
        reducer : callable or dict
            the name of an existing process (e.g. `mean`) or a dict for a
            process graph
        dimension : str, optional
            Defines the dimension to calculate the sum along (defaults to first
            dimension if not specified). Dimensions are expected in this order:
            (dim1, dim2, y, x)
        context: dict, optional
            keyworded parameters needed by the `reducer`

        Returns
        -------
        xr.DataArray


        """

        if callable(reducer):
            return reducer(data, dimension=dimension, **context)
        elif isinstance(reducer, dict):
            # No need to map this
            return data
###############################################################################
# Apply process
###############################################################################


@process
def apply():
    """
    Returns class instance of `Apply`.
    For more details, please have a look at the implementations inside
    `Apply`.

    Returns
    -------
    reduce_dimension :
        Class instance implementing all 'apply' processes.

    """
    return Apply()


class Apply:
    """
    Class implementing all 'apply' processes.

    """

    @staticmethod
    def exec_xar(data, process, context={}):
        """


        Parameters
        ----------
        data : xr.DataArray
            An array of numbers. An empty array resolves always with np.nan.
        process : callable or dict
            the name of an existing process (e.g. `mean`) or a dict for a
            process graph
        context: dict, optional
            keyworded parameters needed by the `reducer`

        Returns
        -------
        xr.DataArray


        """

        if callable(process):
            return process(data, **context)
        return data
###############################################################################
# MergeCubes process
###############################################################################


@process
def merge_cubes():
    """
    Returns class instance of `Merge Cubes`.
    For more details, please have a look at the implementations inside
    `Merge Cubes`.

    Returns
    -------
    merged data cube :
        See the process description for details regarding the dimensions and dimension properties (name, type, labels, reference system and resolution).
    """
    return MergeCubes()


class MergeCubes:
    """
    Class implementing 'merge_cubes' process.

    The data cubes have to be compatible. A merge operation without overlap should be reversible with (a set of) filter operations for each of the two cubes.
    The process performs the join on overlapping dimensions, with the same name and type.
    An overlapping dimension has the same name, type, reference system and resolution in both dimensions, but can have different labels.
    One of the dimensions can have different labels, for all other dimensions the labels must be equal.
    If data overlaps, the parameter overlap_resolver must be specified to resolve the overlap.
    """

    @staticmethod
    def exec_xar(cube1, cube2, overlap_resolver = None, context={}):
        """
        Parameters
        ----------
        cube1 : xr.DataArray
            The first data cube.
        cube2 : xr.DataArray
            The second data cube.
        overlap_resolver : callable or dict or xr.DataArray (created in advance due to mapping)
            the name of an existing process (e.g. `mean`) or a dict for a
            process graph
        context: dict, optional
            keyworded parameters needed by the `reducer`

        Returns
        -------
        xr.DataArray
        """
        if isinstance(cube1, gpd.geodataframe.GeoDataFrame) and isinstance(cube2, gpd.geodataframe.GeoDataFrame):
            if cube1.columns.equals(cube2.columns):
                merged_cube = cube1.append(cube2, ignore_index=True)
                print("Warning - Overlap resolver is not implemented for geopandas vector-cubes, cubes are simply appended!")
            else:
                if 'geometry' in cube1.columns and 'geometry' in cube2.columns and cube1['geometry'].equals(cube2['geometry']):
                    merged_cube = cube1.merge(cube2, on='geometry')
            return merged_cube

        if (cube1.dims == cube2.dims):  # Check if the dimensions have the same names
            matching = 0
            not_matching = 0
            for c in cube1.coords:
                cord = (cube1[c] == cube2[c])  # Check how many dimensions have exactly the same coordinates
                if (np.array([cord.values])).shape[-1] == 1:  # Special case where dimension has only one coordinate, cannot compute len() of that, so I use shape
                    cord = (np.array([cord.values]))  # cord is set 0 or 1 here (False or True)
                else:
                    cord = len(cord)
                if cord == 0:  # dimension with different coordinates
                    dimension = c
                elif cord == (np.array([cube1[c].values])).shape[-1]:  # dimensions with matching coordinates, shape instead of len, for special case with only one coordinate
                    matching += 1
                else:
                    not_matching += 1
                    dim_not_matching = c  # dimensions with some matching coordinates
            if matching + 1 == len(cube1.coords) and not_matching == 0:  # one dimension with different coordinates
                merge = xr.concat([cube1, cube2], dim=dimension)
                merge = merge.sortby(dimension)
            elif matching == len(cube1.coords):  # all dimensions match
                if overlap_resolver is None:  # no overlap resolver, so a new dimension is added
                    merge = xr.concat([cube1, cube2], dim='cubes')
                    merge['cubes'] = ["Cube01", "Cube02"]
                else:
                    if callable(overlap_resolver):  # overlap resolver, for example add
                        merge = overlap_resolver(cube1, cube2, **context)
                    elif isinstance(overlap_resolver, xr.core.dataarray.DataArray):
                        merge = overlap_resolver
                    else:
                        raise Exception('OverlapResolverMissing')
            else:  # WIP
                if not_matching == 1:  # one dimension where some coordinates match, others do not, other dimensions match
                    same1 = []
                    diff1 = []
                    index = 0
                    for t in cube1[dim_not_matching]:  # count matching coordinates
                        if (t == cube2[dim_not_matching]).any():
                            same1.append(index)
                            index += 1
                        else:  # count different coordinates
                            diff1.append(index)
                            index += 1
                    same2 = []
                    diff2 = []
                    index2 = 0
                    for t in cube2[dim_not_matching]:
                        if (t == cube1[dim_not_matching]).any():
                            same2.append(index2)
                            index2 += 1
                        else:
                            diff2.append(index2)
                            index2 += 1
                    if callable(overlap_resolver):
                        c1 = cube1.transpose(dim_not_matching, ...)
                        c2 = cube2.transpose(dim_not_matching, ...)
                        merge = overlap_resolver(c1[same1], c2[same2], **context)
                        if len(diff1) > 0:
                            values_cube1 = c1[diff1]
                            merge = xr.concat([merge, values_cube1], dim=dim_not_matching)
                        if len(diff2) > 0:
                            values_cube2 = c2[diff2]
                            merge = xr.concat([merge, values_cube2], dim=dim_not_matching)
                        merge = merge.sortby(dim_not_matching)
                        merge = merge.transpose(*cube1.dims)
                else:
                    merge = xr.concat([cube1, cube2], dim=dim_not_matching)

        else:  # if dims do not match - WIP
            if len(cube1.dims) < len(cube2.dims):
                c1 = cube1
                c2 = cube2
            else:
                c1 = cube2
                c2 = cube1
            check = []
            for c in c1.dims:
                check.append(c in c1.dims and c in c2.dims)
            for c in c2.dims:
                if not (c in c1.dims):
                    dimension = c
            if np.array(check).all() and len(c2[dimension]) == 1 and callable(overlap_resolver):
                c2 = c2.transpose(dimension, ...)
                merge = overlap_resolver(c2[0], c1, **context)
            elif isinstance(overlap_resolver, xr.core.dataarray.DataArray):
                merge = overlap_resolver
            else:
                raise Exception('OverlapResolverMissing')
        for a in cube1.attrs:
            if a in cube2.attrs and (cube1.attrs[a] == cube2.attrs[a]):
                merge.attrs[a] = cube1.attrs[a]
        return merge


###############################################################################
# FitCurve process
###############################################################################


@process
def fit_curve():
    """
    Returns class instance of `Fit Curve`.
    For more details, please have a look at the implementations inside
    `Fit Curve`.

    Returns
    -------
    fitting parameters for the data cube :
        See the process description for details regarding the dimensions and dimension properties (name, type, labels, reference system and resolution).
    """
    return FitCurve()


class FitCurve:
    """
    Class implementing 'fit_curve' process.

    Use non-linear least squares to fit a model function y = f(x, parameters) to data.
    The process throws an InvalidValues exception if invalid values are encountered.
    Invalid values are finite numbers (see also is_valid).
    """

    @staticmethod
    def exec_xar(data, parameters, function, dimension):
        """
        Parameters
        ----------
        data : xr.DataArray
            A data cube.
        parameters : array
            Defined the number of parameters for the model function and provides an initial guess for them. At least one parameter is required.
        function : child process
            The model function. It must take the parameters to fit as array through the first argument and the independent variable x as the second argument.
            It is recommended to store the model function as a user-defined process on the back-end to be able to re-use the model function with the computed optimal values for the parameters afterwards.
            child process parameters:
            x : number
                The value for the independent variable x.
            parameters : array
                The parameters for the model function, contains at least one parameter.
            Child process return value : number
                The computed value y for the given independent variable x and the parameters.
        dimension : str
            The name of the dimension for curve fitting.
            Must be a dimension with labels that have a order (i.e. numerical labels or a temporal dimension).
            Fails with a DimensionNotAvailable exception if the specified dimension does not exist.
        Returns
        -------
        xr.DataArray
            A data cube with the optimal values for the parameters.
        """
        data = data.fillna(0)  # zero values (masked) are not considered
        if dimension in ['time', 't', 'times']:  # time dimension must be converted into values
            dimension = get_time_dimension_from_data(data, dimension)
            dates = data[dimension].values
            timestep = [((x - np.datetime64('1970-01-01')) / np.timedelta64(1, 's')) for x in dates]
            step = np.array(timestep)
            data[dimension] = step
        else:
            step = dimension

        if isinstance(parameters, xr.core.dataarray.DataArray):
            apply_f = (lambda x, y, p: optimize.curve_fit(function, x[np.nonzero(y)], y[np.nonzero(y)], p)[0])
            in_dims = [[dimension], [dimension], ['params']]
            add_arg = [step, data, parameters]
            output_size = len(parameters['params'])
        else:
            apply_f = (lambda x, y: optimize.curve_fit(function, x[np.nonzero(y)], y[np.nonzero(y)], parameters)[0])
            in_dims = [[dimension], [dimension]]
            add_arg = [step, data]
            output_size = len(parameters)
        values = xr.apply_ufunc(
            apply_f, *add_arg,
            vectorize=True,
            input_core_dims=in_dims,
            output_core_dims=[['params']],
            dask="parallelized",
            output_dtypes=[np.float32],
            dask_gufunc_kwargs={'allow_rechunk': True, 'output_sizes': {'params': output_size}}
        )
        values['params'] = list(range(len(values['params'])))
        values.attrs = data.attrs
        return values

###############################################################################
# PredictCurve process
###############################################################################


@process
def predict_curve():
    """
    Returns class instance of `Predict Curve`.
    For more details, please have a look at the implementations inside
    `Predict Curve`.

    Returns
    -------
    A data cube with the predicted values.
        See the process description for details regarding the dimensions and dimension properties (name, type, labels, reference system and resolution).
    """
    return PredictCurve()


class PredictCurve:
    """
    Class implementing 'predict_curve' process.

    Predict values using a model function and pre-computed parameters.
    """

    @staticmethod
    def exec_xar(data, parameters, function, dimension, labels = None):
        """
        Parameters
        ----------
        data : xr.DataArray
            A data cube to predict values for.
        parameters : xr.DataArray
            A data cube with optimal values from a result of e.g. fit_curve.
        function : child process
            The model function. It must take the parameters to fit as array through the first argument and the independent variable x as the second argument.
            It is recommended to store the model function as a user-defined process on the back-end.
            child process parameters:
            x : number
                The value for the independent variable x.
            parameters : array
                The parameters for the model function, contains at least one parameter.
            Child process return value : number
                The computed value y for the given independent variable x and the parameters.
        dimension : str
            The name of the dimension for curve fitting.
            Must be a dimension with labels that have a order (i.e. numerical labels or a temporal dimension).
            Fails with a DimensionNotAvailable exception if the specified dimension does not exist.
        labels : number, date or date-time
            The labels to predict values for. If no labels are given, predicts values only for no-data (null) values in the data cube.
        Returns
        -------
        xr.DataArray
            A data cube with the predicted values.
        """
        data = data.fillna(0)
        if (np.array([labels])).shape[-1] > 1:
            test = [labels]
        else:
            test = labels
        if dimension in ['time', 't', 'times']:  # time dimension must be converted into values
            dimension = get_time_dimension_from_data(data, dimension)
            dates = data[dimension].values
            if test is None:
                timestep = [((np.datetime64(x) - np.datetime64('1970-01-01')) / np.timedelta64(1, 's')) for x in dates]
                labels = np.array(timestep)
            else:
                coords = labels
                labels = [((np.datetime64(x) - np.datetime64('1970-01-01')) / np.timedelta64(1, 's')) for x in labels]
                labels = np.array(labels)
        else:
            if test is None:
                labels = data[dimension].values
            else:
                coords = labels
        values = xr.apply_ufunc(lambda a: function(labels, *a), parameters,
                                vectorize=True,
                                input_core_dims=[['params']],
                                output_core_dims=[[dimension]],
                                dask="parallelized",
                                output_dtypes=[np.float32],
                                dask_gufunc_kwargs={'allow_rechunk': True, 'output_sizes': {dimension: len(labels)}}
                                )
        if test is None:
            values = values.transpose(*data.dims)
            values[dimension] = data[dimension]
            predicted = data.where(data != 0, values)
        else:
            predicted = values.transpose(*data.dims)
            predicted[dimension] = coords
        if dimension in ['t', 'times']:
            predicted = predicted.rename({dimension: 'time'})
        predicted.attrs = data.attrs
        return predicted


###############################################################################
# Resample cube spatial process
###############################################################################


@process
def resample_cube_spatial():
    """
    Returns class instance of `resample_cube_spatial`.
    For more details, please have a look at the implementations inside
    `resample_cube_spatial`.

    Returns
    -------
    save_result :
        Class instance implementing 'resample_cube_spatial' process.

    """
    return ResampleCubeSpatial()


class ResampleCubeSpatial:
    """
    Class implementing 'resample_cube_spatial' processe.

    """

    @staticmethod
    def exec_xar(data, target, method=None, options={}):
        """
        Save data to disk in specified format.

        Parameters
        ----------
        data : xr.DataArray
           A data cube.
        target: str,
          A data cube that describes the spatial target resolution.
        method: str,
          Resampling method. Methods are inspired by GDAL, see [gdalwarp](https://www.gdal.org/gdalwarp.html) for more information.
          "near","bilinear","cubic","cubicspline","lanczos","average","mode","max","min","med","q1","q3"
          (default: near)

        """
        try:
            methods_list = ["near","bilinear","cubic","cubicspline","lanczos","average","mode","max","min","med","q1","q3"]
            if method is None or method == 'near':
                method = 'nearest'
            elif method not in methods_list:
                raise Exception(f"Selected resampling method \"{method}\" is not available! Please select one of "
                                f"[{', '.join(methods_list)}]")
            return odc.algo._warp.xr_reproject(data,target.geobox,resampling=method)
        except Exception as e:
            raise e


###############################################################################
# Resample cube temporal process
###############################################################################


@process
def resample_cube_temporal():
    """
    Returns class instance of `resample_cube_temporal`.
    For more details, please have a look at the implementations inside
    `resample_cube_temporal`.

    Returns
    -------
    save_result :
        Class instance implementing 'resample_cube_temporal' process.

    """
    return ResampleCubeTemporal()


class ResampleCubeTemporal:
    """
    Class implementing 'resample_cube_temporal' process.

    """

    @staticmethod
    def exec_xar(data, target, dimension = None, valid_within = None):
        """
        Resamples one or more given temporal dimensions from a source data cube to align with the corresponding dimensions of the given target data cube using the nearest neighbor method.
        Returns a new data cube with the resampled dimensions. By default, this process simply takes the nearest neighbor independent of the value (including values such as no-data / null).
        Depending on the data cubes this may lead to values being assigned to two target timestamps.
        To only consider valid values in a specific range around the target timestamps, use the parameter valid_within.
        The rare case of ties is resolved by choosing the earlier timestamps.

        Parameters
        ----------
        data : xr.DataArray
           A data cube.
        target : str,
           A data cube that describes the temporal target resolution.
        dimension : str, null
           The name of the temporal dimension to resample, which must exist with this name in both data cubes.
           If the dimension is not set or is set to null, the process resamples all temporal dimensions that exist with the same names in both data cubes.
           The following exceptions may occur:
           A dimension is given, but it does not exist in any of the data cubes: DimensionNotAvailable
           A dimension is given, but one of them is not temporal: DimensionMismatch
           No specific dimension name is given and there are no temporal dimensions with the same name in the data: DimensionMismatch
        valid_within : number, null
           Setting this parameter to a numerical value enables that the process searches for valid values within the given period of days before and after the target timestamps.
           Valid values are determined based on the function is_valid.
           For example, the limit of 7 for the target timestamps 2020-01-15 12:00:00 looks for a nearest neighbor after 2020-01-08 12:00:00 and before 2020-01-22 12:00:00.
           If no valid value is found within the given period, the value will be set to no-data (null).

        """
        if dimension is None:
            dimension = 'time'
        if dimension in ['time', 't', 'times']:  # time dimension must be converted into values
            dimension = get_time_dimension_from_data(data, dimension)
        else:
            raise Exception('DimensionNotAvailable')
        if dimension not in target.dims:
            target_time = get_time_dimension_from_data(target, dimension)
            target = target.rename({target_time: dimension})
        index = []
        for d in target[dimension].values:
            difference = (np.abs(d - data[dimension].values))
            nearest = np.argwhere(difference == np.min(difference))
            index.append(int(nearest))
        times_at_target_time = data[dimension].values[index]
        new_data = data.loc[{dimension: times_at_target_time}]
        filter_values = new_data[dimension].values
        new_data[dimension] = target[dimension].values
        if valid_within is None:
            new_data = new_data
        else:
            minimum = np.timedelta64(valid_within, 'D')
            filter = (np.abs(filter_values - new_data[dimension].values) <= minimum)
            times_valid = new_data[dimension].values[filter]
            new_data = new_data.loc[{dimension: times_valid}]
        new_data.attrs = data.attrs
        return new_data


###############################################################################
# CreateRasterCube process
###############################################################################


@process
def create_raster_cube():
    """
    Create an empty raster data cube.

    Returns
    -------
    data cube :
        Creates a new raster data cube without dimensions. Dimensions can be added with add_dimension.
    """
    return CreateRasterCube()


class CreateRasterCube:
    """
    Creates a new raster data cube without dimensions. Dimensions can be added with add_dimension.
    """

    @staticmethod
    def exec_num():
        """
        Parameters
        ----------
        This process has no parameters.

        Returns
        -------
        xr.DataArray :
           An empty raster data cube with zero dimensions.
        """
        return xr.DataArray([])


###############################################################################
# AddDimension process
###############################################################################


@process
def add_dimension():
    """
    Adds a new named dimension to the data cube.

    Returns
    -------
    data cube :
        The data cube with a newly added dimension. The new dimension has exactly one dimension label.
        All other dimensions remain unchanged.
    """
    return AddDimension()


class AddDimension:
    """
    Adds a new named dimension to the data cube.
    Afterwards, the dimension can be referred to with the specified name.
    If a dimension with the specified name exists, the process fails with a DimensionExists exception.
    The dimension label of the dimension is set to the specified label.
    """

    @staticmethod
    def exec_xar(data=None, name=None, label=None, type='other'):
        """
        Parameters
        ----------
        data : xr.DataArray
           A data cube to add the dimension to.
        name : str
           Name for the dimension.
        labels : number, str
           A dimension label.
        type : str, optional
           The type of dimension, defaults to other.

        Returns
        -------
        xr.DataArray :
           The data cube with a newly added dimension. The new dimension has exactly one dimension label.
           All other dimensions remain unchanged.
        """
        if name in data.dims:
            raise Exception('DimensionExists - A dimension with the specified name already exists. The existing dimensions are: {}'.format(data.dims))
        data_e = data.assign_coords(placeholder = label)
        data_e = data_e.expand_dims('placeholder')
        data_e = data_e.rename({'placeholder' : name})
        return data_e


###############################################################################
# DimensionLabels process
###############################################################################


@process
def dimension_labels():
    """
    Get the dimension labels.

    Returns
    -------
    Array :
           The labels as an array.

    """
    return DimensionLabels()


class DimensionLabels:
    """
    Gives all labels for a dimension in the data cube. The labels have the same order as in the data cube.
    If a dimension with the specified name does not exist, the process fails with a DimensionNotAvailable exception.
    """

    @staticmethod
    def exec_xar(data, dimension):
        """
        Parameters
        ----------
        data : xr.DataArray
           A data cube to add the dimension to.
        dimension : str
           The name of the dimension to get the labels for.

        Returns
        -------
        np.array :
           The labels as an array.
        """
        if dimension in ['time', 't', 'times']:  # time dimension must be converted into values
            dimension = get_time_dimension_from_data(data, dimension)
            return data[dimension].values
        elif dimension in data.dims:
            return data[dimension].values
        else:
            raise Exception('DimensionNotAvailable')


###############################################################################
# DropDimension process
###############################################################################


@process
def drop_dimension():
    """
    Remove a dimension.

    Returns
    -------
    xr.DataArray :
           A data cube without the specified dimension.

    """
    return DropDimension()


class DropDimension:
    """
    Drops a dimension from the data cube.
    Dropping a dimension only works on dimensions with a single dimension label left,
    otherwise the process fails with a DimensionLabelCountMismatch exception.
    Dimension values can be reduced to a single value with a filter such as filter_bands or the reduce_dimension process.
    If a dimension with the specified name does not exist, the process fails with a DimensionNotAvailable exception.
    """

    @staticmethod
    def exec_xar(data, name):
        """
        Parameters
        ----------
        data : xr.DataArray
           The data cube to drop a dimension from.
        name : str
           Name of the dimension to drop.

        Returns
        -------
        xr.DataArray :
           A data cube without the specified dimension.
           The number of dimensions decreases by one, but the dimension properties
           (name, type, labels, reference system and resolution) for all other dimensions remain unchanged.
        """
        if name in data.dims:
            if len(data[name].values) == 1:
                dropped = data.squeeze(name, drop=True)
                return dropped
            else:
                raise Exception('DimensionLabelCountMismatch')
        else:
            raise Exception('DimensionNotAvailable')

###############################################################################
# RenameDimension process
###############################################################################


@process
def rename_dimension():
    """
    Rename a dimension.

    Returns
    -------
    xr.DataArray :
           A data cube with the same dimensions, but the name of one of the dimensions changes.
           The old name can not be referred to any longer.
           The dimension properties (name, type, labels, reference system and resolution) remain unchanged.
    """
    return RenameDimension()


class RenameDimension:
    """
    Renames a dimension in the data cube while preserving all other properties.
    """

    @staticmethod
    def exec_xar(data, source, target):
        """
        Parameters
        ----------
        data : xr.DataArray
           A data cube.
        source : str
           The current name of the dimension.
           Fails with a DimensionNotAvailable exception if the specified dimension does not exist.
        target : str
           A new Name for the dimension.
           Fails with a DimensionExists exception if a dimension with the specified name exists.

        Returns
        -------
        xr.DataArray :
           A data cube with the same dimensions, but the name of one of the dimensions changes.
           The old name can not be referred to any longer.
           The dimension properties (name, type, labels, reference system and resolution) remain unchanged.
        """
        if source not in data.dims:
            raise Exception('DimensionNotAvailable')
        elif target in data.dims:
            raise Exception('DimensionExists')
        return data.rename({source: target})


###############################################################################
# RenameLabels process
###############################################################################


@process
def rename_labels():
    """
    Rename dimension labels.

    Returns
    -------
    xr.DataArray :
           The data cube with the same dimensions.
           The dimension properties (name, type, labels, reference system and resolution) remain unchanged, except that for the given dimension the labels change.
           The old labels can not be referred to any longer. The number of labels remains the same.
    """
    return RenameLabels()


class RenameLabels:
    """
    Renames a dimension in the data cube while preserving all other properties.
    """

    @staticmethod
    def exec_xar(data, dimension, target, source = []):
        """
        Parameters
        ----------
        data : xr.DataArray
           A data cube.
        dimension : str
           The name of the dimension to rename the labels for.
        target : array
           The new names for the labels. The dimension labels in the data cube are expected to be enumerated if the parameter target is not specified.
           If a target dimension label already exists in the data cube, a LabelExists exception is thrown.
        source : array
           The names of the labels as they are currently in the data cube.
           The array defines an unsorted and potentially incomplete list of labels that should be renamed to the names available in the corresponding array elements in the parameter target.
           If one of the source dimension labels doesn't exist, the LabelNotAvailable exception is thrown.
           By default, the array is empty so that the dimension labels in the data cube are expected to be enumerated.

        Returns
        -------
        xr.DataArray :
           The data cube with the same dimensions.
           The dimension properties (name, type, labels, reference system and resolution) remain unchanged, except that for the given dimension the labels change.
           The old labels can not be referred to any longer. The number of labels remains the same.
        """
        if source == []:
            source = data[dimension].values
        if type(source) in [str, int, float]:
            source = [source]
        if type(target) in [str, int, float]:
            target = [target]
        if len(source) != len(target):
            raise Exception('LabelMismatch')
        source = np.array(source)
        for s in source:
            if s not in data[dimension].values:
                raise Exception('LabelNotAvailable')
        target = np.array(target)
        for t in target:
            if t in data[dimension].values:
                raise Exception('LabelExists')
        names = np.array([])
        for n in data[dimension].values:
            if n in source:
                index = np.argwhere(source == n)
                names = np.append(names, target[int(index)])
            else:
                names = np.append(names, n)
        data[dimension] = names
        return data


###############################################################################
# FilterTemporal process
###############################################################################


@process
def filter_temporal():
    """
    Temporal filter for a temporal intervals.

    Returns
    -------
    xr.DataArray :
           A data cube restricted to the specified temporal extent.
           The dimensions and dimension properties (name, type, labels, reference system and resolution) remain unchanged,
           except that the temporal dimensions (determined by dimensions parameter) may have less dimension labels.

    """
    return FilterTemporal()


class FilterTemporal:
    """
    Limits the data cube to the specified interval of dates and/or times.
    More precisely, the filter checks whether each of the temporal dimension labels is greater than or equal to the lower boundary (start date/time)
    and less than the value of the upper boundary (end date/time).
    This corresponds to a left-closed interval, which contains the lower boundary but not the upper boundary.
    """

    @staticmethod
    def exec_xar(data, extent, dimension = None):
        """
        Parameters
        ----------
        data : xr.DataArray
           A data cube.
        extent : temporal interval, array
           Left-closed temporal interval, i.e. an array with exactly two elements:
           The first element is the start of the temporal interval. The specified instance in time is included in the interval.
           The second element is the end of the temporal interval. The specified instance in time is excluded from the interval.
           The specified temporal strings follow RFC 3339. Also supports open intervals by setting one of the boundaries to null, but never both.
        dimension : str
           The name of the temporal dimension to filter on.
           If no specific dimension is specified or it is set to null, the filter applies to all temporal dimensions.
           Fails with a DimensionNotAvailable exception if the specified dimension does not exist.

        Returns
        -------
        xr.DataArray :
           A data cube restricted to the specified temporal extent.
           The dimensions and dimension properties (name, type, labels, reference system and resolution) remain unchanged,
           except that the temporal dimensions (determined by dimensions parameter) may have less dimension labels.
        """
        if dimension is None:
            dimension = 'time'
        dimension = get_time_dimension_from_data(data, dimension)
        if 'Z' in extent[0]:
            extent[0] = extent[0][:-1]
        if 'Z' in extent[1]:
            extent[1] = extent[1][:-1]
        if dimension in data.dims:
            filtered = data.loc[{dimension: slice(np.datetime64(extent[0]), np.datetime64(extent[1]))}]
            l = np.min([len(str(filtered[dimension].values[-1])), len(str(extent[1]))])
            if str(filtered[dimension].values[-1])[:l] == str(extent[1])[:l]:
                skip_last = filtered[dimension].values[:-1]
                return filtered.loc[{dimension: skip_last}]
            else:
                return filtered
        else:
            raise Exception('DimensionNotAvailable')

###############################################################################
# FilterSpatial process
###############################################################################


@process
def filter_spatial():
    """
    Spatial filter using geometries.

    Returns
    -------
    xr.DataArray :
           A data cube restricted to the specified geometries. The dimensions and dimension properties
           (name, type, labels, reference system and resolution) remain unchanged, except that the spatial dimensions
           have less (or the same) dimension labels.

    """
    return FilterSpatial()


class FilterSpatial:
    """
    Limits the data cube over the spatial dimensions to the specified geometries.
    For polygons, the filter retains a pixel in the data cube if the point at the pixel center intersects with at least
    one of the polygons (as defined in the Simple Features standard by the OGC).
    For points, the process considers the closest pixel center.
    For lines (line strings), the process considers all the pixels whose centers are closest to at least one point on
    the line. More specifically, pixels outside of the bounding box of the given geometry will not be available after
    filtering. All pixels inside the bounding box that are not retained will be set to null (no data).
    """

    @staticmethod
    def exec_xar(data, geometries):
        """
        Parameters
        ----------
        data : xr.DataArray
           A data cube.
        geometries : geojson
           One or more geometries used for filtering, specified as GeoJSON.

        Returns
        -------
        xr.DataArray :
           A data cube restricted to the specified geometries. The dimensions and dimension properties
           (name, type, labels, reference system and resolution) remain unchanged, except that the spatial dimensions
           have less (or the same) dimension labels.
        """
        lon = data['x'].values
        lat = data['y'].values

        if geometries['type'] == 'Point':
            coord = geometries['coordinates']
            arg = np.argmin(abs(lon - coord[0]))
            x = lon[arg]
            arg = np.argmin(abs(lat - coord[1]))
            y = lat[arg]
            return data.loc[{'x': x, 'y': y}]

        elif geometries['type'] == 'LineString' or geometries['type'] == 'Lines' or geometries['type'] == 'Polygon':
            coord = geometries['coordinates']
            if geometries['type'] == 'Polygon':
                P = Polygon(coord)
            else:
                P = LineString(coord)
            coord = np.array(coord)
            lon = lon[lon >= np.min(coord[:, 0])]
            lon = lon[lon <= np.max(coord[:, 0])]
            lat = lat[lat >= np.min(coord[:, 1])]
            lat = lat[lat <= np.max(coord[:, 1])]
            d2 = data.loc[{'x': lon, 'y': lat}]

        elif geometries['type'] == 'MultiPolygon':
            coord = geometries['coordinates']
            polygons = [Polygon(x) for x in coord]
            P = MultiPolygon(polygons)
            coord = np.array(coord)
            lon = lon[lon >= np.min(coord[:, :, 0])]
            lon = lon[lon <= np.max(coord[:, :, 0])]
            lat = lat[lat >= np.min(coord[:, :, 1])]
            lat = lat[lat <= np.max(coord[:, :, 1])]
            d2 = data.loc[{'x': lon, 'y': lat}]

        def insides(x, y):
            y0 = y[0]
            val = np.array([])
            for i in range(len(x)):
                x0 = x[i]
                p = Point(x0, y0)
                val = np.append(val, p.within(P))
            return val

        if 'time' in d2.dims:
            D = d2.isel(time=0)
            if 'bands' in D.dims:
                D = D.isel(bands=0)
        elif 'bands' in d2.dims:
            D = d2.isel(bands=0)
        else:
            D = d2
        D = D.where(np.isnan(D), D['y'])
        values = xr.apply_ufunc(
            insides, D['x'].values, D,
            vectorize=True,
            input_core_dims=[['x'], ['x']],
            output_core_dims=[['x']],
            dask="parallelized",
            output_dtypes=[np.float32],
            dask_gufunc_kwargs={'allow_rechunk': True}
        )
        return d2.where(values == 1, np.nan)

########################################################################################################################
# Filter Labels Process
########################################################################################################################

@process
def filter_labels():
    """
    Filter dimension labels based on a condition.

    Returns
    -------
    xr.DataArray :
           A data cube with the same dimensions. The dimension properties (name, type, labels, reference system and
           resolution) remain unchanged, except that the given dimension has less (or the same) dimension labels.

    """
    return FilterLabels()


class FilterLabels:
    """
    Filters the dimension labels in the data cube for the given dimension. Only the dimension labels that match the
    specified condition are preserved, all other labels with their corresponding data get removed.
    """

    @staticmethod
    def exec_xar(data, condition, dimension, context = {}):
        """
        Parameters
        ----------
        data : xr.DataArray
           A data cube.
        condition : process
           A condition that is evaluated against each dimension label in the specified dimension. A dimension label and
           the corresponding data is preserved for the given dimension, if the condition returns true.
        dimension : str
           The name of the dimension to filter on. Fails with a DimensionNotAvailable exception if the specified
           dimension does not exist.
        context : any
           Additional data to be passed to the condition.

        Returns
        -------
        xr.DataArray :
           A data cube with the same dimensions. The dimension properties (name, type, labels, reference system and
           resolution) remain unchanged, except that the given dimension has less (or the same) dimension labels.
        """
        if dimension in ['time', 't', 'times']:
            dimension = get_time_dimension_from_data(data, dimension)
        if dimension not in data.dims:
            raise DimensionNotAvailable("A dimension with the specified name does not exist.")
        labels = data[dimension].values
        if callable(condition):
            label_mask = condition(labels, **context)
            label = labels[label_mask]
            data_f = data.loc[{dimension: label}]
            return data_f


########################################################################################################################
# Filter Bbox Process
########################################################################################################################

@process
def filter_bbox():
    """
    Spatial filter using a bounding box.

    Returns
    -------
    xr.DataArray :
           A data cube restricted to the bounding box. The dimensions and dimension properties (name, type, labels,
           reference system and resolution) remain unchanged, except that the spatial dimensions have less (or the
           same) dimension labels.

    """
    return FilterBbox()


class FilterBbox:
    """
    Limits the data cube to the specified bounding box.
    The filter retains a pixel in the data cube if the point at the pixel center intersects with the bounding box (as
    defined in the Simple Features standard by the OGC).
    """

    @staticmethod
    def exec_xar(data, extent):
        """
        Parameters
        ----------
        data : xr.DataArray
           A data cube.
        extent : bounding-box:object
           A bounding box, which may include a vertical axis (see base and height).

        Returns
        -------
        xr.DataArray :
           A data cube restricted to the bounding box. The dimensions and dimension properties (name, type, labels,
           reference system and resolution) remain unchanged, except that the spatial dimensions have less (or the
           same) dimension labels.
        """
        if type(extent) == dict:
            if "crs" in extent and extent["crs"] is not None:
                crs = extent["crs"]
            else:
                crs = 4326
            crs_input = CRS.from_user_input(crs)

            if "west" in extent and "east" in extent and "south" in extent and "north" in extent:
                bbox = [[extent["south"], extent["west"]],
                        [extent["south"], extent["east"]],
                        [extent["north"], extent["east"]],
                        [extent["north"], extent["west"]]]
            else:
                raise Exception("Coordinate missing!")
            if "crs" in data.attrs:
                data_crs = data.attrs["crs"]
                crs_data = CRS.from_user_input(data_crs)

            transformer = Transformer.from_crs(crs_input, crs_data)

            x_t = []
            y_t = []
            for p in bbox:
                x1, y1 = p
                x2, y2 = transformer.transform(x1, y1)
                x_t.append(x2)
                y_t.append(y2)
            x_t = np.array(x_t)
            y_t = np.array(y_t)
            x_min = x_t.min()
            x_max = x_t.max()
            y_min = y_t.min()
            y_max = y_t.max()

            data_x = data['x'].values
            data_x = data_x[data_x > x_min][data_x[data_x > x_min] < x_max]
            data = data.sel(x=data_x)
            data_y = data['y'].values
            data_y = data_y[data_y > y_min][data_y[data_y > y_min] < y_max]
            data = data.sel(y= data_y)
        return data


########################################################################################################################
# Mask Process
########################################################################################################################

@process
def mask():
    """
    Returns class instance of `Mask`.
    For more details, please have a look at the implementations inside `Mask`.

    Returns
    -------
    Mask :
        Class instance implementing all 'mask' processes.

    """
    return Mask()


class Mask:
    """
    Class instance implementing all 'mask' processes.

    """

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, mask, replacement=np.nan):
        """
        Applies a mask to an array. A mask is an array for which corresponding elements among `data` and `mask` are
        compared and those elements in `data` are replaced whose elements in `mask` are non-zero (for numbers) or True
        (for boolean values). The elements are replaced with the value specified for `replacement`, which defaults to
        np.nan (no data).


        Parameters
        ----------
        data : np.ndarray
            An array to mask.
        mask : np.ndarray
            A mask as an array. Every element in `data` must have a corresponding element in `mask`.
        replacement : float or int, optional
            The value used to replace masked values with.
        dimension : int, optional
            Defines the dimension along to apply the mask (default is 0).

        Returns
        -------
        np.ndarray :
            The masked array.

        """
        if data.shape == mask.shape:
            data[mask] = replacement
        else:
            data[mask[None, :, :]] = replacement

        return data

    @staticmethod
    def exec_xar(data, mask, replacement=np.nan):
        """
        Applies a mask to an array. A mask is an array for which corresponding elements among `data` and `mask` are
        compared and those elements in `data` are replaced whose elements in `mask` are non-zero (for numbers) or True
        (for boolean values). The elements are replaced with the value specified for `replacement`, which defaults to
        np.nan (no data).


        Parameters
        ----------
        data : xr.DataArray
            An array to mask.
        mask : xr.DataArray
            A mask as an array. Every element in `data` must have a corresponding element in `mask`.
        replacement : float or int, optional
            The value used to replace masked values with.

        Returns
        -------
        xr.DataArray :
            The masked array.
        """
        if mask.dtype != bool:
            mask = mask != 0
        if (data.dims == mask.dims):  # Check if the dimensions have the same names
            matching = 0
            not_matching = 0
            for c in data.coords:
                cord = (data[c] == mask[c])  # Check how many dimensions have exactly the same coordinates
                if (np.array([cord.values])).shape[-1] == 1:  # Special case where dimension has only one coordinate, cannot compute len() of that, so I use shape
                    cord = (np.array([cord.values]))  # cord is set 0 or 1 here (False or True)
                else:
                    cord = len(cord)
                if cord == 0:  # dimension with different coordinates
                    dimension = c
                elif cord == (np.array([data[c].values])).shape[-1]:  # dimensions with matching coordinates, shape instead of len, for special case with only one coordinate
                    matching += 1
                else:
                    not_matching += 1
                    dim_not_matching = c  # dimensions with some matching coordinates
            if matching != len(data.coords):  # all dimensions match
                raise Exception(
                    "The following dimension does not match between data and mask: %s, can't apply the process.\ndata has %d and mask has %d samples." % (
                    str(dim_not_matching), len(data[dim_not_matching]), len(mask[dim_not_matching])))
            data = data.where(mask == 0, replacement)
        elif len(data.dims) > len(mask.dims):
            data = data.where(mask == 0, replacement)
        else:
            raise Exception("The mask has more dimensions than the data, can't apply the process.")
            pass
        return data


########################################################################################################################
# Mask_Polygon Process
########################################################################################################################

@process
def mask_polygon():
    """
    Returns class instance of `Mask Polygon`.
    For more details, please have a look at the implementations inside `Mask Polygon`.

    Returns
    -------
    Mask :
        Apply a polygon mask

    """
    return Mask_Polygon()


class Mask_Polygon:
    """
    Apply a polygon mask

    """
    @staticmethod
    def exec_xar(data, mask, replacement=np.nan, inside=False):
        """
        Applies a (multi) polygon mask to a raster data cube. To apply a raster mask use mask.
        All pixels for which the point at the pixel center does not intersect with any polygon (as defined in the
        Simple Features standard by the OGC) are replaced. This behavior can be inverted by setting the parameter
        inside to true. The pixel values are replaced with the value specified for replacement, which defaults to null
        (no data). No data values in data will be left untouched by the masking operation.


        Parameters
        ----------
        data : xr.DataArray
            An array to mask.
        mask : geojson
            A GeoJSON object containing at least one polygon. The provided feature types can be one of the following:
            A Polygon or MultiPolygon geometry,
        replacement : float or int, optional
            The value used to replace masked values with.
        inside : boolean
            If set to true all pixels for which the point at the pixel center does intersect with any polygon are
            replaced.

        Returns
        -------
        xr.DataArray :
            A masked raster data cube with the same dimensions.
            The dimension properties (name, type, labels, reference system and resolution) remain unchanged.
        """

        def insides(x, y):
            y0 = y[0]
            val = np.array([])
            for i in range(len(x)):
                x0 = x[i]
                p = Point(x0, y0)
                val = np.append(val, p.within(polygon))
            return val

        if mask['type'] == 'Point':
            lon = data['x'].values
            lat = data['y'].values
            coord = mask['coordinates']
            arg = np.argmin(abs(lon - coord[0]))
            x = lon[arg]
            arg = np.argmin(abs(lat - coord[1]))
            y = lat[arg]
            if inside:
                return data.where(data.x != x, replacement).where(data.y != y, replacement)
            else:
                return data.where(data.x == x, replacement).where(data.y == y, replacement)
        elif mask['type'] == 'Polygon':
            coord = mask['coordinates']
            polygon = Polygon(coord)

        elif mask['type'] == 'MultiPolygon':
            coord = mask['coordinates']
            polygons = [Polygon(x) for x in coord]
            polygon = MultiPolygon(polygons)

        if 'time' in data.dims:
            D = data.isel(time=0)
            if 'bands' in D.dims:
                D = D.isel(bands=0)
        elif 'bands' in data.dims:
            D = data.isel(bands=0)
        else:
            D = data
        D = D.fillna(1)
        D = D.where(np.isnan(D), D['y'])
        values = xr.apply_ufunc(
            insides, D['x'].values, D,
            vectorize=True,
            input_core_dims=[['x'], ['x']],
            output_core_dims=[['x']],
            dask="parallelized",
            output_dtypes=[np.float32],
            dask_gufunc_kwargs={'allow_rechunk': True}
        )
        if inside:
            return data.where(values == 1, replacement)
        else:
            return data.where(values == 0, replacement)



###############################################################################
# AggregateTemporalPeriod process
###############################################################################


@process
def aggregate_temporal_period():
    """
    Temporal aggregations based on calendar hierarchies

    Returns
    -------
    xr.DataArray :
           A new data cube with the same dimensions. The dimension properties
           (name, type, labels, reference system and resolution) remain unchanged, except for the resolution and
           dimension labels of the given temporal dimension.
    """
    return AggregateTemporalPeriod()


class AggregateTemporalPeriod:
    """
    Computes a temporal aggregation based on calendar hierarchies such as years, months or seasons.
    For other calendar hierarchies aggregate_temporal can be used.
    For each interval, all data along the dimension will be passed through the reducer.
    If the dimension is not set or is set to null, the data cube is expected to only have one temporal dimension.
    """

    @staticmethod
    def exec_xar(data, period, reducer, dimension=None, context=None):
        """
        Parameters
        ----------
        data : xr.DataArray
            An array to mask.
        period : string
            The time intervals to aggregate. The following pre-defined values are available:
            hour, day, week, dekad, month, season, tropical-season, year, decade, decade-ad
        reducer : process
            A reducer to be applied for the values contained in each period.
            A reducer is a single process such as mean or a set of processes,
            which computes a single value for a list of values, see the category 'reducer' for such processes.
            Periods may not contain any values, which for most reducers leads to no-data (null) values by default.
        dimension : string
            The name of the temporal dimension for aggregation.
            All data along the dimension is passed through the specified reducer.
            If the dimension is not set or set to null, the data cube is expected to only have one temporal dimension.
            Fails with a TooManyDimensions exception if it has more dimensions.
            Fails with a DimensionNotAvailable exception if the specified dimension does not exist.
        context : any
            Additional data to be passed to the reducer.

        Returns
        -------
        xr.DataArray :
           A new data cube with the same dimensions. The dimension properties
           (name, type, labels, reference system and resolution) remain unchanged, except for the resolution and
           dimension labels of the given temporal dimension.
        """
        if dimension is None:
            dimension = 'time'
        if dimension in ['time', 't', 'times']:
            dimension = get_time_dimension_from_data(data, dimension)
        else:
            raise Exception('DimensionNotAvailable')
        if period in ['hour', 'day', 'week', 'month', 'season', 'year']:
            p = dimension + '.' + period
        if context is None:
            context = {'dimension': dimension}
        elif 'dimension' not in context:
            context['dimension'] = dimension
        if callable(reducer):
            new = reducer(data.groupby(p), **context)
        return new

###############################################################################
# ApplyDimension process
###############################################################################

@process
def apply_dimension():
    """
    Apply a process to pixels along a dimension

    Returns
    -------
    xr.DataArray :
           A data cube with the newly computed values.
           All dimensions stay the same, except for the dimensions specified in corresponding parameters.
    """
    return ApplyDimension()

class ApplyDimension:
    """
    Applies a process to all pixel values along a dimension of a raster data cube.
    For example, if the temporal dimension is specified the process will work on a time series of pixel values.
    The process reduce_dimension also applies a process to pixel values along a dimension, but drops the dimension
    afterwards. The process apply applies a process to each pixel value in the data cube. The target dimension is the
    source dimension if not specified otherwise in the target_dimension parameter. The pixel values in the target
    dimension get replaced by the computed pixel values. The name, type and reference system are preserved. The
    dimension labels are preserved when the target dimension is the source dimension and the number of pixel values in
    the source dimension is equal to the number of values computed by the process. Otherwise, the dimension labels will
    be incrementing integers starting from zero, which can be changed using rename_labels afterwards. The number of
    labels will equal to the number of values computed by the process.
    """

    @staticmethod
    def exec_xar(data, process, dimension, target_dimension=None, context=None):
        """
        Parameters
        ----------
        data : xr.DataArray
            A data cube..
        process : Process
            Process to be applied on all pixel values.
            The specified process needs to accept an array and must return an array with at least one element.
            A process may consist of multiple sub-processes.
        dimension : string
            The name of the source dimension to apply the process on.
            Fails with a DimensionNotAvailable exception if the specified dimension does not exist.
        target_dimension : string
            The name of the target dimension or null (the default) to use the source dimension specified in the
            parameter dimension. By specifying a target dimension, the source dimension is removed. The target
            dimension with the specified name and the type other (see add_dimension) is created, if it doesn't exist
            yet.
        context : any
            Additional data to be passed to the process.

        Returns
        -------
        xr.DataArray :
           A data cube with the newly computed values.
           All dimensions stay the same, except for the dimensions specified in corresponding parameters.
        """
        if callable(process):
            context = context if context is not None else {}
            if dimension in ['time', 't', 'times']:
                dimension = get_time_dimension_from_data(data, dimension)
            new = process(data, dimension=dimension, **context)
            if dimension in new.dims:
                if target_dimension is not None:
                    new = new.rename({dimension: target_dimension})
            else:
                if target_dimension is not None:
                    new = new.expand_dims(target_dimension)
                    new[target_dimension] = np.arange(len(new[target_dimension]))
            return new
        else:
            return process

###############################################################################
# Aggregate_spatial process
###############################################################################

@process
def aggregate_spatial():

    return AggregateSpatial()


class AggregateSpatial:

    @staticmethod
    def exec_xar(data, geometries, reducer, target_dimension="result", context=None):
        if len(data.dims) > 3:
            raise TooManyDimensions(f'The number of dimensions must be reduced to three for aggregate_spatial. Input raster-cube dimensions: {data.dims}')

        if isinstance(geometries, gpd.geodataframe.GeoDataFrame):
            if not geometries.crs:
                geometries = geometries.set_crs(DEFAULT_CRS)

        # If a geopandas.GeoDataFrame is provided make sure it has a crs set
        else:
            # If raw geojson is provided, construct a gpd.geodataframe.GeoDataFrame from that
            try:

                # Each feature must have a properties field, even if there is no property
                # This is necessary due to this bug in geopandas: https://github.com/geopandas/geopandas/pull/2243
                for feature in geometries['features']:
                    if 'properties' not in feature:
                        feature['properties'] = {}
                    elif feature['properties'] is None:
                        feature['properties'] = {}
                
                geometries_crs = geometries.get('crs', DEFAULT_CRS) 
                geometries = gpd.GeoDataFrame.from_features(geometries, crs=geometries_crs)
            except:
                raise Exception('[!] No compatible vector input data has been provided.')

        ## Input geometries are in EPSG:4326 and the data has a different projection. We reproject the vector-cube to fit the data.
        if 'crs' in data.attrs:
            data_crs = data.attrs['crs']
        else:
            data_crs = 'PROJCS["Azimuthal_Equidistant",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Azimuthal_Equidistant"],PARAMETER["latitude_of_center",53],PARAMETER["longitude_of_center",24],PARAMETER["false_easting",5837287.81977],PARAMETER["false_northing",2121415.69617],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
        data = data.rio.set_crs(data_crs)
        
        vector_cube_utm = geometries.to_crs(data_crs)

        # This is to make sure the geopandas GeoDataFrame index enumerates the geometries starting from 0 to len(geometries) 
        vector_cube_utm = vector_cube_utm.reset_index(drop=True)

        # Add mask column
        crop_list = []
        valid_count_list = []
        total_count_list = []

        ## Loop over the geometries in the FeatureCollection
        for _, row in vector_cube_utm.iterrows():
            # rasterise geometry to create mask. This will 
            mask = make_geocube(gpd.GeoDataFrame({ 'geometry': [row["geometry"]], "mask": [1] }), measurements=["mask"], like=data)
            geom_crop = data.where(mask.mask == 1).stack(dimensions={"flattened": ["x", "y"]}).reset_index('flattened')
            crop_list.append(geom_crop)

            total_count = geom_crop.count(dim="flattened")
            valid_count = geom_crop.where(~geom_crop.isnull()).count(dim="flattened")
            valid_count_list.append(valid_count)
            total_count_list.append(total_count)

        try:
            # Make sure that NaN values are ignored
            reducer = partial(reducer, ignore_nodata=True)
        except TypeError:
            pass

        # Reduce operation
        xr_crop_list = xr.concat(crop_list, "result")
        xr_crop_list_reduced = reducer(xr_crop_list, dimension="flattened")
        xr_crop_list_reduced_dropped = xr_crop_list_reduced.drop(["spatial_ref"])
        xr_crop_list_reduced_dropped_ddf = xr_crop_list_reduced_dropped.to_dataset(dim="bands").to_dask_dataframe().drop("result", axis=1)
        output_ddf_merged = xr_crop_list_reduced_dropped_ddf.merge(vector_cube_utm)

        # Metadata gathering operation
        valid_count_list_xr = xr.concat(valid_count_list, dim="valid_count").drop("spatial_ref")
        total_count_list_xr = xr.concat(total_count_list, dim="total_count").drop("spatial_ref")
        valid_count_list_xr_ddf = valid_count_list_xr.to_dataset(dim="bands").to_dask_dataframe().drop("valid_count", axis=1).add_suffix("_valid_count")
        total_count_list_xr_ddf = total_count_list_xr.to_dataset(dim="bands").to_dask_dataframe().drop("total_count", axis=1).add_suffix("_total_count")

        # Merge all these dataframes
        output_vector_cube = dask.dataframe.concat([output_ddf_merged, valid_count_list_xr_ddf, total_count_list_xr_ddf], axis=1)

        # turn the output back in to a dask-geopandas GeoDataFrame
        output_vector_cube_ddf = dask_geopandas.from_dask_dataframe(output_vector_cube)
        output_vector_cube_ddf = output_vector_cube_ddf.set_crs(data_crs)

        return output_vector_cube_ddf


###############################################################################
# FitRegrRandomForest process
###############################################################################

@process
def fit_regr_random_forest():

    return FitRegrRandomForest()


class FitRegrRandomForest:

    @staticmethod
    def exec_xar(predictors, target, num_trees = 100, max_variables = None, seed = None, predictors_vars = None, target_var = None, client = None):
        CHUNK_SIZE_ROWS = 1500
        
        params = {
            'learning_rate': 1,
            'max_depth': 5,
            'num_parallel_tree': num_trees,
            'objective': 'reg:squarederror',
            'subsample': 0.8,
            'tree_method': 'hist'}

        # TODO: This needs to be fixed to accept the number of columns, rather than a fraction
        if max_variables is not None:
            params['colsample_bynode'] = max_variables

        if isinstance(predictors, dict):
            predictors = gpd.GeoDataFrame.from_features(predictors)
        elif isinstance(predictors, str):
            predictors = gpd.GeoDataFrame.from_file(predictors)

        if isinstance(target, dict):
            target = gpd.GeoDataFrame.from_features(target)
        elif isinstance(target, str):
            target = gpd.GeoDataFrame.from_file(target)

        if isinstance(predictors, gpd.geodataframe.GeoDataFrame) and isinstance(target, gpd.geodataframe.GeoDataFrame):
            if not predictors['geometry'].equals(target['geometry']):
                raise Exception('Geometries of input predictors and target do not match.')
            predictors_pandas = pd.DataFrame(predictors)
            predictors_pandas = predictors_pandas.drop(columns=['geometry'])
            target_pandas = pd.DataFrame(target)
            target_pandas = target_pandas.drop(columns=['geometry'])
            X = df.from_pandas(predictors_pandas, chunksize=CHUNK_SIZE_ROWS)
            for c in X.columns:
                if c not in predictors_vars:
                    X = X.drop(columns=c)

            y = df.from_pandas(target_pandas, chunksize=CHUNK_SIZE_ROWS)
            for c in y.columns:
                if c != target_var:
                    y = y.drop(columns=c)

            dtrain = xgb.dask.DaskDMatrix(client, X, y)

            output = xgb.dask.train(client, params, dtrain, num_boost_round=1)

            return output
        else:
            raise Exception('[!] No compatible vector input data has been provided.')


###############################################################################
# PredictRandomForest process
###############################################################################

@process
def predict_random_forest():

    return PredictRandomForest()


class PredictRandomForest:

    @staticmethod
    def exec_xar(data, model, dimension, predictors_vars = None, client = None):
        if dimension in ['time', 't', 'times']:  # time dimension must be converted into values
            dimension = get_time_dimension_from_data(data, dimension)
        predictor_cols = list(data.dims)
        if dimension in predictor_cols:
            predictor_cols.remove(dimension)
        if len(predictor_cols) == 1:
            stacked = data
            I = stacked[predictor_cols[0]].values
        else:
            stacked = data.stack(z=predictor_cols)
            I = stacked['z'].values
            predictor_cols = ['z']

        X_hat = stacked.to_dataset(dim=dimension).to_dask_dataframe().drop(columns=predictor_cols)
        if 'spatial_ref' in X_hat:
            X_hat = X_hat.drop(columns=['spatial_ref'])
        if predictors_vars is not None:
            for c in X_hat.columns:
                if c not in predictors_vars:
                    X_hat = X_hat.drop(columns=c)

        y_hat = xgb.dask.predict(client, model, X_hat).to_frame()
        y_hat_ds = xarray_dataset_from_dask_dataframe(y_hat)
        y_hat_da = y_hat_ds.to_array(dim=dimension)
        y_hat_da['index'] = I
        y_hat_da = y_hat_da.rename({'index': predictor_cols[0]})
        p = data.loc[{dimension: data[dimension].values[0]}]
        predictions_xr = xr.ones_like(p) * y_hat_da
        predictions_xr.attrs = data.attrs
        predictions_xr[dimension] = np.array(['prediction'])
        return predictions_xr


###############################################################################
# SaveMLModel process
###############################################################################

@process
def save_ml_model():

    return SaveMLModel()


class SaveMLModel:

    @staticmethod
    def exec_num(model, output_filepath = 'path'):
        if isinstance(model, dict) and 'booster' in model:
            m = model['booster']
        elif isinstance(model, xgb.core.Booster):
            m = model
        else:
            pass
        path = f'{output_filepath}/out_model.json'
        with open(f'{output_filepath}/product.yml', 'a') as product_file:
            product_file.close()
        m.save_model(path)


###############################################################################
# LoadMLModel process
###############################################################################

@process
def load_ml_model():

    return LoadMLModel()


class LoadMLModel:

    @staticmethod
    def exec_num(model, input_filepath = 'path'):
        date = os.listdir(f'{input_filepath}/jobs/{model}')
        if len(date) > 0:
            date = date[0]
        filepath = f'{input_filepath}/jobs/{model}/{date}/result/out_model.json'
        model_xgb = xgb.Booster()
        model_xgb.load_model(filepath)
        return model_xgb

###############################################################################
# FlattenDimensions process
###############################################################################

@process
def flatten_dimensions():

    return FlattenDimensions()


class FlattenDimensions:

    @staticmethod
    def exec_xar(data, dimensions, target_dimension, label_seperator='~'):
        for i in range(len(dimensions)):
            if dimensions[i] in ['time', 't', 'times']:  # time dimension must be converted into values
                t = get_time_dimension_from_data(data, dimensions[i])
                dimensions[i] = t
        stacked = data.stack({target_dimension:(tuple(dimensions))})
        return stacked


###############################################################################
# UnflattenDimensions process
###############################################################################

@process
def unflatten_dimension():

    return UnflattenDimension()


class UnflattenDimension:

    @staticmethod
    def exec_xar(data, dimensions=None, target_dimension=None, label_seperator='~'):
        stacked = data.unstack()
        return stacked

@process
def load_vector_cube():

    return LoadVectorCube()


class LoadVectorCube:

    @staticmethod
    def exec_num(job_id=None, URL=None, input_filepath='path'):
        if not (job_id or URL):
            raise Exception("One of these parameters needs to be provided: <job_id>, <URL>")
        
        # TODO: Loading random files from untrusted URLs is dangerous, this has to be rethought going forward! 
        if URL:
            try:
                response = urllib.request.urlopen(URL)
                geometries = json.loads(response.read())
            except json.JSONDecodeError:
                raise Exception('[!] Unable to parse vector-data from provided URL.')
            except:
                raise Exception('[!] Unable to load vector-data from provided URL.')

        if job_id:
            result_files = os.listdir(f'{input_filepath}/jobs/{job_id}')
            if len(result_files) > 0:
                latest_date = result_files[0]
            filepath = f'{input_filepath}/jobs/{job_id}/{latest_date}/result/out_vector_cube.json'
            f = open(filepath)
            geometries = json.load(f)

        # Each feature must have a properties field, even if there is no property
        # This is necessary due to this bug in geopandas: https://github.com/geopandas/geopandas/pull/2243
        for feature in geometries['features']:
            if 'properties' not in feature:
                feature['properties'] = {}
            elif feature['properties'] is None:
                feature['properties'] = {}
        
        geometries_crs = geometries.get('crs', {}).get("properties", {}).get("name", DEFAULT_CRS)
        
        try:
            gdf = gpd.GeoDataFrame.from_features(geometries, crs=geometries_crs)
        except:
            raise Exception('[!] No compatible vector input data has been provided.')

        return gdf

@process
def save_vector_cube():

    return SaveVectorCube()


class SaveVectorCube:

    """Note that at this stage, this process assumes that each job can only save a single model, using the job-id as an ID for the resulting file."""
    @staticmethod
    def exec_xar(data, output_filepath = 'path'):
        path = f'{output_filepath}/out_vector_cube.json'
        with open(f'{output_filepath}/product.yml', 'a') as product_file:
            product_file.close()

        if isinstance(data, gpd.GeoDataFrame):
            data_gpd = data
        else:
            try:
                data_gpd = gpd.GeoDataFrame.from_features(data)
            except Exception:
                raise TypeError("Couldn't transform the provided vector-cube to a geopandas.Geodataframe. \
                    For the `data` arg, please provide a .geojson or a geopandas.Geodataframe.")
            

        data_gpd.to_file(path, driver='GeoJSON')
