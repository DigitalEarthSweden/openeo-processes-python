import functools
import re
from datetime import timezone, timedelta, datetime
from typing import Any, Callable, Tuple

import numpy as np
import xarray as xr
import geopandas as gpd


def eval_datatype(data):
    """
    Returns a data type tag depending on the data type of `data`.
    This can be:
        - "numpy": `nump.ndarray`
        - "xarray": `xarray.DataArray`
        - "dask": `dask.array.core.Array`
        - "int", "float", "dict", "list", "set", "tuple", "NoneType": Python builtins
        - "datetime": `datetime.datetime`
        - "function": callable object

    Parameters
    ----------
    data : object
        Data to get the data type from.

    Returns
    -------
    str :
        Data type tag.

    """
    package = type(data).__module__
    package_root = package.split(".", 1)[0]
    if package in ("builtins", "datetime"):
        return type(data).__name__
    elif package_root in ("numpy", "xarray", "dask", "datacube", "geopandas"):
        return package_root
    else:
        return package + '.' + type(data).__name__


def tuple_args_to_np_array(args, kwargs) -> np.array:
    np_args = [np.array(arg) for arg in args if isinstance(arg, tuple)]
    np_kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, tuple):
            np_kwargs[key] = np.array(value)
        else:
            np_kwargs[key] = value
    return np_args, np_kwargs


# Registry of processes (dict mapping process id to wrapped process implementation).
_processes = {}


def process(processor):
    """
    This function serves as a decorator for empty openEO process definitions, which call a class `processor` defining
    the process implementations for different data types.

    Parameters
    ----------
    processor : class
        Class implementing an openEO process containing the methods `exec_num`, `exec_np`, `exec_xar`, or `exec_dar`.

    Returns
    -------
    object :
        Process/function wrapper returning the result of the process.

    """
    @functools.wraps(processor)
    def fun_wrapper(*args, **kwargs):
        cls = processor()

        # Workaround to allow mapping correctly also List(xr.DataArray)
        # TODO: remove automatic conversion from List to np.Array and update all tests
        # Convert lists to numpy arrays
        datatypes = None
        if args:
            # Check if there is a list of xr.DataArrays in the first variable
            if isinstance(args[0], list) and np.any(tuple(True if isinstance(a, xr.DataArray) else False for a in args[0])):
                datatypes = ["xarray"]
            else:
                args = tuple(list2nparray(a) if isinstance(a, list) else a for a in args)
        if kwargs:
            # Check if there is a list of xr.DataArrays in variable 'data'
            if 'data' in kwargs and isinstance(kwargs['data'], list) and np.any(tuple(True if isinstance(a, xr.DataArray) else False for a in kwargs['data'])):
                datatypes = ["xarray"]
            else:
                kwargs = {k: (list2nparray(v) if isinstance(v, list) else v) for k, v in kwargs.items()}
        if not datatypes:
            # retrieve data types of input (keyword) arguments
            datatypes = set(eval_datatype(a) for a in args)
            datatypes.update(eval_datatype(v) for v in kwargs.values())

        datatypes = set(datatypes)
        if "datacube" in datatypes:
            cls_fun = getattr(cls, "exec_odc")
        elif datatypes.intersection(["xarray", "dask", "geopandas", "xgboost"]):
            cls_fun = getattr(cls, "exec_xar")
        elif "numpy" in datatypes:
            cls_fun = getattr(cls, "exec_np")
        elif datatypes.issubset({"int", "float", "NoneType", "str", "bool", "datetime", "dict"}):
            cls_fun = getattr(cls, "exec_num")
        elif "tuple" in datatypes:
            args, kwargs = tuple_args_to_np_array(args, kwargs)
            cls_fun = getattr(cls, "exec_np")
        else:
            raise Exception('Datatype unknown.')

        return cls_fun(*args, **kwargs)

    process_id = processor.__name__.rstrip('_')
    _processes[process_id] = fun_wrapper

    return fun_wrapper


def has_process(process_id: str) -> bool:
    """
    Check if the given process is defined

    Parameters
    ----------
    process_id : str
           Process id

    Returns
    -------
    True if the process is defined, False otherwise
    """
    return process_id in _processes


def get_process(process_id: str) -> Callable:
    """
    Get the function corresponding with given process id

    Parameters
    ----------
    process_id : str
           Process id

    Returns
    -------
    Python function (callable) that wraps the process
    """
    return _processes[process_id]





def list2nparray(x):
    """
    Converts a list in a nump

    Parameters
    ----------
    x : list or np.ndarray
        List to convert.

    Returns
    -------
    np.ndarray

    """
    x_tmp = np.array(x)
    if x_tmp.dtype.kind in ['U', 'S']:
        x = np.array(x, dtype=object)
    else:
        x = x_tmp

    return x


def create_slices(index, axis=0, n_axes=1):
    """
    Creates a multidimensional slice index.

    Parameters
    ----------
    index : int
        The zero-based index of the element to retrieve (default is 0).
    axis : int, optional
        Axis of the given index (default is 0).
    n_axes : int, optional
        Number of axes (default is 1).

    Returns
    -------
    tuple of slice:
        Tuple of index slices.

    """

    slices = [slice(None)] * n_axes
    slices[axis] = index

    return tuple(slices)


def str2time(string, allow_24h=False):
    """
    Converts time strings in various formats to a datetime object.
    The datetime formats follow the RFC3339 convention.

    Parameters
    ----------
    string : str
        String representation of time or date.
    allow_24h : bool, optional
        If True, `string` is allowed to contain '24' as hour value.

    Returns
    -------
    datetime.datetime :
        Parsed datetime object.

    """

    # handle timezone formatting and replace possibly occuring ":" in time zone string
    # handle timezone formatting for +
    if "+" in string:
        string_parts = string.split('+')
        string_parts[-1] = string_parts[-1].replace(':', '')
        string = "+".join(string_parts)

    # handle timezone formatting for -
    if "t" in string.lower():  # a full datetime string is given
        time_string = string[10:]
        if "-" in time_string:
            string_parts = time_string.split('-')
            string_parts[-1] = string_parts[-1].replace(':', '')
            string = string[:10] + "-".join(string_parts)
    else:  # a time string is given
        if "-" in string:
            string_parts = string.split('-')
            string_parts[-1] = string_parts[-1].replace(':', '')
            string = "-".join(string_parts)

    # searches for 24 in hour value
    pattern = re.compile("24:\d{2}:\d{2}")
    pattern_match = re.search(pattern, string)
    if pattern_match:
        if allow_24h:  # if the user allows 24 as an hour value, replace 24 by 23 and add a timedelta of one hour later
            old_sub_string = pattern_match.group()
            new_sub_string = "23" + old_sub_string[2:]
            string = string.replace(old_sub_string, new_sub_string)
        else:
            err_msg = "24 is not allowed as an hour value. Hours are only allowed to be given in the range 0 - 23. " \
                      "Set 'allow_24h' to 'True' if you want to translate 24 as a an hour."
            raise ValueError(err_msg)

    rfc3339_time_formats = ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S.%f",
                            "%Y-%m-%dT%H:%M:%Sz", "%Y-%m-%dt%H:%M:%SZ", "%Y-%m-%dt%H:%M:%Sz", "%Y-%m-%dT%H:%M:%S%z",
                            "%Y-%m-%dt%H:%M:%S%z", "%H:%M:%SZ", "%H:%M:%S%z"]
    date_time = None
    # loops through each format and takes the one for which the translation succeeded first
    for i, used_time_format in enumerate(rfc3339_time_formats):
        try:
            date_time = datetime.strptime(string, used_time_format)
            if date_time.tzinfo is None:
                date_time = date_time.replace(tzinfo=timezone.utc)
            break
        except:
            continue

    # add a timedelta of one hour if 24 is allowed as an hour value
    if date_time and allow_24h:
        date_time += timedelta(hours=1)

    return date_time


def get_time_dimension_from_data(data: xr.DataArray, dim: str = "time") -> str:
    """Automatically find the time dimension available in the dataset.

    Support 't' and 'time' (OpenEO preferres 't', internally 'time' is used
    """
    if dim in data.dims:
        return dim
    time_dimensions = ["time", "t", "times"]
    for time_dim in time_dimensions:
        if time_dim in data.dims:
            return time_dim

def keep_attrs(x, y, data):
    """Keeps the attributes of the inputs x and y in the output data.

    When a processes, which requires two inputs x and y is used,
    the attributes of x and y are not forwarded to the output data.
    This checks if one of the inputs is a Dataarray, which has attributes.
    The attributes of x or y are then given to the output data.
    """
    if isinstance(x, xr.DataArray) and isinstance(y, xr.DataArray):
        for a in x.attrs:
            if a in y.attrs and (x.attrs[a] == y.attrs[a]):
                data.attrs[a] = x.attrs[a]
    elif isinstance(x, xr.DataArray):
        data.attrs = x.attrs
    elif isinstance(y, xr.DataArray):
        data.attrs = y.attrs
    return data

def xarray_dataset_from_dask_dataframe(dataframe):
    """Utility function snatched from @AyrtonB at https://github.com/pydata/xarray/pull/4659.
    
    Convert a dask.dataframe.DataFrame into an xarray.Dataset
    This method will produce a Dataset from a dask DataFrame.
    Dimensions are loaded into memory but the data itself remains
    a dask array.
    Parameters
    ----------
    dataframe : dask.dataframe.DataFrame
        Dask DataFrame from which to copy data and index.
    Returns
    -------
    Dataset
        The converted Dataset
    See also
    --------
    xarray.DataArray.from_dask_series
    xarray.Dataset.from_dataframe
    xarray.DataArray.from_series
    """
    import dask.dataframe as dd

    if not dataframe.columns.is_unique:
        raise ValueError("cannot convert DataFrame with non-unique columns")
    if not isinstance(dataframe, dd.DataFrame):
        raise ValueError("cannot convert non-dask dataframe objects")

    idx = dataframe.index.compute()

    arrays = [(k, v.to_dask_array(lengths=True)) for k, v in dataframe.items()]

    obj = xr.Dataset()
    index_name = idx.name if idx.name is not None else "index"
    dims = (index_name,)
    obj[index_name] = (dims, idx)

    for name, values in arrays:
        obj[name] = (dims, values)

    return obj

if __name__ == '__main__':
    pass
