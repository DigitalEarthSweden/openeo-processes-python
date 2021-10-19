from datetime import datetime
from os.path import splitext
from typing import Any, Dict, List

import numpy as np
import odc.algo
import rioxarray  # needed by save_result even if not directly called
import xarray as xr
from openeo_processes.extension.odc import write_odc_product
from openeo_processes.utils import process, get_time_dimension_from_data
from scipy import optimize

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
    def exec_odc(odc_cube, product: str, dask_chunks: dict):

        odc_params = {
            'product': product,
            'dask_chunks': dask_chunks
        }
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
        Save data to disk in specified format.

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
        def reformat_dataset(dataset: xr.DataArray, timestamp: datetime = None, has_time_dim: bool = False) \
                -> xr.Dataset:
            """Create a xarray Dataset."""
            data_var = dataset.fillna(-9999)
            data_var.attrs["nodata"] = -9999
            if 'bands' in dataset.coords:
                try:
                    tmp = data_var.to_dataset(dim="bands")
                except Exception as e:
                    print(e)
                    n = dataset['bands'].values
                    tmp = data_var.to_dataset(name=str(n))
            else:
                tmp = data_var.to_dataset(name='result')

            # fix dimension order
            current_dims = tuple(tmp.dims)
            base_dims = ("bands", "time", "y", "x") if has_time_dim else ("bands", "y", "x")
            additional_dim = list(set(current_dims).difference(set(base_dims)))
            base_dims = base_dims[1:]  # remove bands elem > not a dimension!
            if additional_dim and current_dims != (additional_dim[0], *base_dims):
                tmp = tmp.transpose(additional_dim[0], *base_dims)
            elif current_dims != base_dims:
                tmp = tmp.transpose(*base_dims)

            tmp.attrs = data_var.attrs
            # This is a hack! ODC always(!) expectes to have a time dimension
            # set datetime to now if no other information is available
            if has_time_dim:
                tmp = tmp.rename({"time": "t"})
            else:
                tmp.attrs["datetime_from_dim"] = str(timestamp) if timestamp else str(datetime.now())
            if "crs" not in tmp.attrs:
                first_data_var = tmp.data_vars[list(tmp.data_vars.keys())[0]]
                tmp.attrs["crs"] = first_data_var.geobox.crs.to_wkt()
            return tmp

        def refactor_data(data: xr.DataArray) -> List[xr.Dataset]:
            """Recreate a Dataset from the final result as Dataarray, to ensure a well formatted netCDF."""
            all_tmp = []
            # TODO this must be improved once `rename_dimension` is supported!
            if 'time' in data.coords:
                for timestamp in data.time.values:
                    data_at_timestamp = data.loc[dict(time=timestamp)]
                    all_tmp.append(reformat_dataset(data_at_timestamp, timestamp))
            else:
                all_tmp.append(reformat_dataset(data))

            return all_tmp

        def create_output_filepath(output_filepath: str, idx: int = 0, ext: str = "nc") -> str:
            """Create the output filepath."""
            root, _ = splitext(output_filepath)
            return f'{root}_{str(idx).zfill(5)}.{ext}'

        # start workaround
        # https://github.com/opendatacube/datacube-core/issues/972
        # ValueError failed to prevent overwriting existing key units in `attrs` on variable 'time'
        if hasattr(data, 'time') and hasattr(data.time, 'units'):
            data.time.attrs.pop('units', None)
        if hasattr(data, 'grid_mapping'):
            data.attrs.pop('grid_mapping')
        # end workaround

        dask_result = data.compute()  # perform computation
        formats = ('GTiff', 'netCDF')
        if format.lower() == 'netcdf':
            data_list = refactor_data(dask_result)
            paths = []
            for idx in range(len(data_list)):
                paths.append(create_output_filepath(output_filepath, idx, 'nc'))
            # combined dataset
            if 'time' in dask_result.coords:
                combined_dataset = reformat_dataset(dask_result, None, has_time_dim=True)
                data_list.append(combined_dataset)
                combined_path = f'{splitext(output_filepath)[0]}_combined.nc'
                paths.append(combined_path)
            xr.save_mfdataset(data_list, paths)

        elif format.lower() in ['gtiff','geotiff']:
            data_list = refactor_data(dask_result)
            if len(data_list[0].dims) > 3:
                raise Exception("[!] Not possible to write a 4-dimensional GeoTiff, use NetCDF instead.")
            for idx, dataset in enumerate(data_list):
                cur_output_filepath = create_output_filepath(output_filepath, idx, 'tif')
                dataset.rio.to_raster(raster_path=cur_output_filepath,**options)

        else:
            raise ValueError(f"Error when saving to file. Format '{format}' is not in {formats}.")

        if write_prod:
            write_odc_product(data_list[0], output_filepath)

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
    def exec_xar(data, target, method, options={}):
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
        if len(data[dimension].values) >= len(target[dimension].values):
            index = np.array([])
            for d in data[dimension].values:
                difference = (np.abs(d - target[dimension].values))
                nearest = np.argwhere(difference == np.min(difference))
                index = np.append(index, nearest)
            t = []
            for i in index:
                t.append(target[dimension].values[int(i)])
            filter_values = data[dimension].values
            new_data = data #ATTENTION new_data is a shallow copy of data! When you change new_data also data is changed.
            new_data[dimension] = t
        else:
            index = []
            for d in target[dimension].values:
                difference = (np.abs(d - data[dimension].values))
                nearest = np.argwhere(difference == np.min(difference))
                index.append(int(nearest))
            data_t = data.transpose(dimension, ...)
            new_data = data_t[index]
            new_data = new_data.transpose(*data.dims)
            filter_values = new_data[dimension].values
            new_data[dimension] = target[dimension].values
        if valid_within is None:
            new_data = new_data
        else:
            minimum = np.timedelta64(valid_within, 'D')
            filter = (np.abs(filter_values - new_data[dimension].values) <= minimum)
            new_data_t = new_data.transpose(dimension, ...)
            new_data_t = new_data_t[filter]
            new_data = new_data_t.transpose(*new_data.dims)
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
    def exec_xar(data, name, labels, type = 'other'):
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
        data_e = data.assign_coords(placeholder = labels)
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
