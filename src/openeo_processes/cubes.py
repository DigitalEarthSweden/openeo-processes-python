import rioxarray  # needed by save_result even if not directly called
from openeo_processes.utils import process
from os.path import splitext
import xarray as xr
import odc.algo
import numpy as np
import pandas as pd
from scipy import optimize

###############################################################################
# Load Collection Process
###############################################################################


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
        return load_result(odc_cube, product, dask_chunks, x, y, time,
                           measurements, crs)


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

        datacube = odc_cube.load(**odc_params)
        # Convert to xr.DataArray
        # TODO: add conversion with multiple and custom dimensions
        datacube = datacube.to_array(dim='bands')

        return datacube


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
        elif isinstance(process, dict):
            # No need to map this
            return data


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
        data = data.fillna(0) # zero values (masked) are not considered
        if dimension in ['time', 't', 'times']:  # time dimension must be converted into values
            dates = data[dimension].values
            timestep = [((x - np.datetime64('1970-01-01')) / np.timedelta64(1, 's')) for x in dates]
            step = np.array(timestep)
            data[dimension] = step
        else:
            step = dimension
        param = len(optimize.curve_fit(function, step, step * 0)[0])  # how many parameters are calculated
        if not len([parameters]) == param:  # how many input parameters are given
            parameters = np.ones(param) * parameters[-1]
        values = xr.apply_ufunc(lambda x, y: optimize.curve_fit(function, x[np.nonzero(y)], y[np.nonzero(y)], parameters)[0], step, data, # zero values not considered
                                vectorize=True,
                                input_core_dims=[[dimension], [dimension]],  # Dimension along we fit the curve function
                                output_core_dims=[['params']],
                                dask="parallelized",
                                output_dtypes=[np.float32],
                                dask_gufunc_kwargs={'allow_rechunk': True, 'output_sizes': {'params': param}}
                                )
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
            dates = data[dimension].values
            if test == None:
                timestep = [((x - np.datetime64('1970-01-01')) / np.timedelta64(1, 's')) for x in dates]
                labels = np.array(timestep)
            else:
                coords = labels
                labels = [((x - np.datetime64('1970-01-01')) / np.timedelta64(1, 's')) for x in labels]
                labels = np.array(labels)
        else:
            if test == None:
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
        if test == None:
            values = values.transpose(*data.dims)
            predicted = xr.DataArray(values, coords=data.coords, dims=data.dims)
            predicted = predicted.where(data==0, data)
        else:
            predicted = values.transpose(*data.dims)
            predicted[dimension] = coords
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
    Class implementing 'save_result' processe.

    """

    @staticmethod
    def exec_xar(data, output_filepath='out', format='GTiff', options={}):
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
        
        def refactor_data(data):
            # The following code is required to recreate a Dataset from the final result as Dataarray, to get a well formatted netCDF
            if 'time' in data.coords:
                tmp = xr.Dataset(coords={'t':data.time.values,'y':data.y,'x':data.x})
                if 'bands' in data.coords:
                    try:
                        for var in data['bands'].values:
                            tmp[str(var)] = (('t','y','x'),data.loc[dict(bands=var)])
                    except Exception as e:
                        print(e)
                        tmp[str((data['bands'].values))] = (('t','y','x'),data)
                else:
                    tmp['result'] = (('t','y','x'),data)
            else:
                tmp = xr.Dataset(coords={'y':data.y,'x':data.x})
                if 'bands' in data.coords:
                    try:
                        for var in data['bands'].values:
                            tmp[str(var)] = (('y','x'),data.loc[dict(bands=var)])
                    except Exception as e:
                        print(e)
                        tmp[str((data['bands'].values))] = (('y','x'),data)
                else:
                    tmp['result'] = (('y','x'),data)
            tmp.attrs = data.attrs
            return tmp
        
        formats = ('GTiff', 'netCDF')
        if format.lower() == 'netcdf':
            if not splitext(output_filepath)[1]:
                output_filepath = output_filepath + '.nc'
            # start workaround
            # https://github.com/opendatacube/datacube-core/issues/972
            # ValueError failed to prevent overwriting existing key units in `attrs` on variable 'time'
            if hasattr(data, 'time') and hasattr(data.time, 'units'):
                data.time.attrs.pop('units', None)
            # end workaround
            
            data = refactor_data(data)
            data.to_netcdf(path=output_filepath)

        elif format.lower() in ['gtiff','geotiff']:
            if not splitext(output_filepath)[1]:
                output_filepath = output_filepath + '.tif'
            # TODO
            # Add check, this works only for 2D or 3D DataArrays, else loop is needed
            
            data = refactor_data(data)
            if len(data.dims) > 3:
                if len(data.t)==1:
                    # We keep the time variable as band in the GeoTiff, multiple band/variables of the same timestamp
                    data = data.squeeze('t')
                else:
                    raise Exception("[!] Not possible to write a 4-dimensional GeoTiff, use NetCDF instead.")
            
            data.rio.to_raster(raster_path=output_filepath,**options)

            
        else:
            raise ValueError(f"Error when saving to file. Format '{format}' is not in {formats}.")

            
###############################################################################
# Resample cube temporal process
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
