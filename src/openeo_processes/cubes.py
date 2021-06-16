
import rioxarray  # needed by save_result even if not directly called
from openeo_processes.utils import process
from os.path import splitext

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
    def exec_odc(odc_cube, product: str, x: tuple, y: tuple, time: list,
                 dask_chunks: dict, measurements: list = [],
                 crs: str = "EPSG:4326"):

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

        elif format.lower() in ['gtiff','gtif','geotiff','geotif','tiff','tif']:
            if not splitext(output_filepath)[1]:
                output_filepath = output_filepath + '.tif'
            # TODO
            # Add check, this works only for 2D or 3D DataArrays, else loop is needed
            
            data = refactor_data(data)
            if len(data.dims) > 3:
                if len(data.t)==1:
                    # We keep the time variable as band in the GeoTiff, multiple band/variables of the same timestamp
                    data = data.squeeze('time')
                else:
                    raise Exception("[!] Not possible to write a 4-dimensional GeoTiff, use NetCDF instead.")
            
            data.rio.to_raster(raster_path=output_filepath,**options)

            
        else:
            raise ValueError(f"Error when saving to file. Format '{format}' is not in {formats}.")
