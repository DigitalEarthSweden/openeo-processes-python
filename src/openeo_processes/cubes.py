import rioxarray  # needed by save_result even if not directly called
from openeo_processes.utils import process
from os.path import splitext
import xarray as xr
from scipy import optimize
import odc.algo
import numpy as np

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
        overlap_resolver : callable or dict
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
                if overlap_resolver == None:  # no overlap resolver, so a new dimension is added
                    values = np.array([cube1.values, cube2.values])
                    cubes = ["Cube01", "Cube02"]
                    coords = [cubes]
                    dimensions = ["cubes"]
                    for d in cube1.dims:
                        dimensions.append(d)
                        coords.append(cube1[d])
                    merge = xr.DataArray(values, coords=coords, dims=dimensions)
                else:
                    if callable(overlap_resolver):  # overlap resolver, for example add
                        values = overlap_resolver(cube1, cube2, **context)
                        merge = xr.DataArray(values, coords=cube1.coords,
                                             dims=cube1.dims)  # define dimensions like in cube1
                    else:
                        raise Exception(overlap_resolver, 'not found!')
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
            dims_l = c2.dims
            for c in c1.dims:
                check.append(c in c1.dims and c in c2.dims)
            for c in c2.dims:
                if not (c in c1.dims):
                    dimension = c
            c2 = c2.transpose(dimension, ...)
            length = len(c2[dimension])
            if np.array(check).all():
                if callable(overlap_resolver):  # overlap resolver, for example add
                    values = []
                    for l in range(length):
                        values.append(overlap_resolver(c2[l], c1, **context).values)
                    merge = xr.DataArray(values, coords=c2.coords,
                                         dims=c2.dims)  # define dimensions like in larger cube
                    merge = merge.transpose(*dims_l)
            else:
                raise Exception(overlap_resolver, 'not found!')

        return merge

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
