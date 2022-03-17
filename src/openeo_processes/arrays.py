import numpy as np
import pandas as pd
import xarray as xr
import math

from openeo_processes.utils import create_slices
from openeo_processes.utils import process
from openeo_processes.comparison import is_valid
from openeo_processes.comparison import is_empty

from openeo_processes.errors import ArrayElementNotAvailable
from openeo_processes.errors import ArrayElementParameterMissing
from openeo_processes.errors import ArrayElementParameterConflict
from openeo_processes.errors import GenericError

try:
    from xarray_extras.sort import topk, argtopk
except ImportError:
    topk = None
    argtopk = None


########################################################################################################################
# Array Create Process
########################################################################################################################

@process
def array_create():
    """
    Returns class instance of `ArrayCreate`.
    For more details, please have a look at the implementations inside `ArrayCreate`.

    Returns
    -------
    ArrayCreate :
        Class instance implementing all 'array_create' processes.

    """
    return ArrayCreate()


class ArrayCreate:
    """
    Class implementing all 'array_create' processes.

    """
    @staticmethod
    def exec_num(data, repeat = 1):
        """
        Creates a new array, which by default is empty.
        The second parameter repeat allows to add the given array multiple times to the new array.
        In most cases you can simply pass a (native) array to processes directly, but this process is especially useful
        to create a new array that is getting returned by a child process, for example in apply_dimension.

        Parameters
        ----------
        data : int
            A (native) array to fill the newly created array with. Defaults to an empty array.
        repeat : int
            The number of times the (native) array specified in data is repeatedly added after each other to the new
            array being created. Defaults to 1.

        Returns
        -------
        np.array :
            The newly created array.

        """
        data = np.array([data])
        if len(data) == 0:
            return np.array([])
        return np.tile(data, reps=repeat)


    @staticmethod
    def exec_np(data = [], repeat = 1):
        """
        Creates a new array, which by default is empty.
        The second parameter repeat allows to add the given array multiple times to the new array.
        In most cases you can simply pass a (native) array to processes directly, but this process is especially useful
        to create a new array that is getting returned by a child process, for example in apply_dimension.

        Parameters
        ----------
        data : np.array
            A (native) array to fill the newly created array with. Defaults to an empty array.
        repeat : int
            The number of times the (native) array specified in data is repeatedly added after each other to the new
            array being created. Defaults to 1.

        Returns
        -------
        np.array :
            The newly created array.

        """
        data = np.array(data)
        if len(data) == 0:
            return np.array([])
        return np.tile(data, reps=repeat)

    @staticmethod
    def exec_xar(data, repeat = 1):
        """
        Creates a new array, which by default is empty.
        The second parameter repeat allows to add the given array multiple times to the new array.
        In most cases you can simply pass a (native) array to processes directly, but this process is especially useful
        to create a new array that is getting returned by a child process, for example in apply_dimension.

        Parameters
        ----------
        data : xr.DataArray
            A (native) array to fill the newly created array with. Defaults to an empty array.
        repeat : int
            The number of times the (native) array specified in data is repeatedly added after each other to the new
            array being created. Defaults to 1.

        Returns
        -------
        np.array :
            The newly created array.

        """
        if len(data) == 0:
            return np.array([])
        elif len(data.shape) == 1:
            if len(data) < 100:
                data = data.values
                return np.tile(data, reps=repeat)

    @staticmethod
    def exec_da():
        pass

########################################################################################################################
# Array Modify Process
########################################################################################################################

@process
def array_modify():
    """
    Returns class instance of `ArrayModify`.
    For more details, please have a look at the implementations inside `ArrayModify`.

    Returns
    -------
    ArrayModify :
        Class instance implementing all 'array_modify' processes.

    """
    return ArrayModify()


class ArrayModify:
    """
    Class implementing all 'array_modify' processes.

    """

    @staticmethod
    def exec_num():
        pass


    @staticmethod
    def exec_np(data, values, index, length = 1):
        """
        Modify an array by removing, inserting or updating elements. Updating can be seen as removing elements followed
        by inserting new elements (not necessarily the same number). All labels get discarded and the array indices are
        always a sequence of numbers with the step size of 1 and starting at 0

        Parameters
        ----------
        data : np.array
            The array to modify.
        values : np.array
            The values to insert into the data array.
        index : int
            The index in the data array of the element to insert the value(s) before. If the index is greater than the
            number of elements in the data array, the process throws an ArrayElementNotAvailable exception.
            To insert after the last element, there are two options:
            Use the simpler processes array_append to append a single value or array_concat to append multiple values.
            Specify the number of elements in the array. You can retrieve the number of elements with the process
            count, having the parameter condition set to true.
        lenght : int
            The number of elements in the data array to remove (or replace) starting from the given index.
            If the array contains fewer elements, the process simply removes all elements up to the end.

        Returns
        -------
        np.array :
            An array with values added, updated or removed.

        """
        data = np.array(data)
        values = np.array(values)
        if index == 0:
            part = values
        else:
            first = data[:index]
            part = np.append(first, values)
        if index+length < len(data):
            part = np.append(part, data[index+length:])
        return part


    @staticmethod
    def exec_xar():
        pass

    @staticmethod
    def exec_da():
        pass

########################################################################################################################
# Array Concat Process
########################################################################################################################

@process
def array_concat():
    """
    Returns class instance of `ArrayConcat`.
    For more details, please have a look at the implementations inside `ArrayConcat`.

    Returns
    -------
    ArrayModify :
        Class instance implementing all 'array_concat' processes.

    """
    return ArrayConcat()


class ArrayConcat:
    """
    Class implementing all 'array_concat' processes.

    """

    @staticmethod
    def exec_num():
        pass


    @staticmethod
    def exec_np(array1, array2):
        """
        Concatenates two arrays into a single array by appending the second array to the first array. Array labels get
        discarded from both arrays before merging.

        Parameters
        ----------
        array1 : np.array
            The first array.
        array2 : np.array
            The second array.

        Returns
        -------
        np.array :
            The merged array.

        """
        array1 = np.array(array1)
        array2 = np.array(array2)
        concat = np.append(array1, array2)
        return concat


    @staticmethod
    def exec_xar():
        pass

    @staticmethod
    def exec_da():
        pass


########################################################################################################################
# Array Contains Process
########################################################################################################################

@process
def array_contains():
    """
    Returns class instance of `ArrayContains`.
    For more details, please have a look at the implementations inside `ArrayContains`.

    Returns
    -------
    ArrayContains :
        Class instance implementing all 'array_contains' processes.

    """
    return ArrayContains()


class ArrayContains:
    """
    Class implementing all 'array_contains' processes.

    """

    @staticmethod
    def exec_num():
        pass

    # TODO: refine this implementation for larger arrays
    @staticmethod
    def exec_np(data, value):
        """
        Checks whether the array specified for `data` contains the value specified in `value`.
        Returns `True` if there's a match, otherwise `False`.

        Parameters
        ----------
        data : np.array
            Array to find the value in.
        value : object
            Value to find in `data`.

        Returns
        -------
        bool :
            Returns `True` if the list contains the value, `False` otherwise.

        Notes
        -----
        `in` is not working because this process checks only for the first level.

        """
        for elem in data:
            if np.array(pd.isnull(value)).all() and np.isnan(elem):  # special handling for nan values
                return True
            elif np.array(elem == value).all():
                return True
        return False

    @staticmethod
    def exec_xar(data, value):
        """
        Checks whether the array specified for `data` contains the value specified in `value`.
        Returns `True` if there's a match, otherwise `False`.

        Parameters
        ----------
        data : xr.DataArray
            Array to find the value in.
        value : object
            Value to find in `data`.

        Returns
        -------
        bool :
            Returns `True` if the list contains the value, `False` otherwise.

        Notes
        -----
        `in` is not working because this process checks only for the first level.
        """
        if np.array(value).size == 1:
            if pd.isnull(value):
                return (data.isnull().sum().values > 0)
            else:
                return data.isin(value).sum().values > 0
        else:
            return data.isin(value) #TODO: Check what should happen when value contains more than one value

    @staticmethod
    def exec_da():
        pass


########################################################################################################################
# Array Element Process
########################################################################################################################

@process
def array_element():
    """
    Returns class instance of `ArrayElement`.
    For more details, please have a look at the implementations inside `ArrayElement`.

    Returns
    -------
    ArrayElement :
        Class instance implementing all 'array_element' processes.

    """
    return ArrayElement()


class ArrayElement:
    """
    Class implementing all 'array_element' processes.

    """

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, index=None, label=None, return_nodata=False, dimension=0, labels=None):
        """
        Returns the element with the specified index or label from the array. Either the parameter `index` or `label`
        must be specified, otherwise the `ArrayElementParameterMissing` exception is thrown. If both parameters are set
        the `ArrayElementParameterConflict` exception is thrown.

        Parameters
        ----------
        data : np.array
            An array.
        index : int, optional
            The zero-based index of the element to retrieve (default is 0).
        label : int or str, optional
            The label of the element to retrieve.
        dimension : int, optional
            Defines the index dimension (default is 0).
        return_nodata : bool, optional
            By default this process throws an `ArrayElementNotAvailable` exception if the index or label is invalid.
            If you want to return np.nan instead, set this flag to `True`.
        labels : np.array, optional
            The available labels. This is needed when specifing `label`.

        Returns
        -------
        object
            The value of the requested element.

        Raises
        ------
        ArrayElementNotAvailable :
            The array has no element with the specified index or label.
        ArrayElementParameterMissing :
            Either `index` or `labels` must be set.
        ArrayElementParameterConflict :
            Only `index` or `labels` allowed to be set.

        """

        ArrayElement._check_input(index, label)
        if label and (labels is None):
            msg = "Parameter 'labels' is needed when specifying input parameter 'label'."
            raise GenericError(msg)
        if label:
            # Convert label to index, using labels
            index = labels.tolist().index(label)
        if index >= data.shape[dimension]:
            if not return_nodata:
                raise ArrayElementNotAvailable()
            else:
                array_elem = np.nan
        else:
            idx = create_slices(index, axis=dimension, n_axes=len(data.shape))
            array_elem = data[idx]

        return array_elem

    @staticmethod
    def exec_xar(data, dimension, index=None, label=None, return_nodata=False, labels=None):
        """
        Returns the element with the specified index or label from the array. Either the parameter `index` or `label`
        must be specified, otherwise the `ArrayElementParameterMissing` exception is thrown. If both parameters are set
        the `ArrayElementParameterConflict` exception is thrown.

        Parameters
        ----------
        data : xr.DataArray
            An xarray DataArray object.
        dimension : str
            Defines the dimension name.
        index : int, optional
            The zero-based index of the element to retrieve (default is 0).
        label : str, optional
            The label of the element to retrieve.
        return_nodata : bool, optional
            By default this process throws an `ArrayElementNotAvailable` exception if the index or label is invalid.
            If you want to return np.nan instead, set this flag to `True`.
        labels : np.array, optional
            The available labels. This is needed when specifing `label`.

        Returns
        -------
        object
            The value of the requested element.

        Raises
        ------
        ArrayElementNotAvailable :
            The array has no element with the specified index or label.
        ArrayElementParameterMissing :
            Either `index` or `labels` must be set.
        ArrayElementParameterConflict :
            Only `index` or `labels` allowed to be set.

        """

        ArrayElement._check_input(index, label)
        if label:
            try:
                array_elem = data.loc[{dimension: label}]
            except KeyError:
                raise ArrayElementNotAvailable()

        if index is not None:  # index could be 0
            try:
                array_elem = data[{dimension: index}]
            except IndexError:
                raise ArrayElementNotAvailable()

        if dimension not in array_elem.dims:
            # Drop coord for 'dimension' if only one value was left
            # in this dim
            array_elem = array_elem.drop_vars(dimension)

        return array_elem

    @staticmethod
    def exec_da():
        pass

    @staticmethod
    def _check_input(index, label): #, labels=None):
        """
        Checks if `index` and `label` are given correctly.

        Either the parameter `index` or `label` must be specified, otherwise the `ArrayElementParameterMissing`
        exception is thrown. If both parameters are set the `ArrayElementParameterConflict `exception is thrown.

        Parameters
        ----------
        index : int, optional
            The zero-based index of the element to retrieve (default is 0).
        label : int or str, optional
            The label of the element to retrieve.
        # labels : np.array, optional
        #     The available labels.

        Raises
        ------
        ArrayElementParameterMissing :
            Either `index` or `labels` must be set.
        ArrayElementParameterConflict :
            Only `index` or `labels` allowed to be set.

        """
        if (index is not None) and (label is not None):
            raise ArrayElementParameterConflict()

        if index is None and label is None:
            raise ArrayElementParameterMissing()
        
        # if label and labels is None:
        #     msg = "Parameter 'labels' is needed when specifying input parameter 'label'."
        #     raise GenericError(msg)


########################################################################################################################
# Count Process
########################################################################################################################

@process
def count():
    """
    Returns class instance of `Count`.
    For more details, please have a look at the implementations inside `Count`.

    Returns
    -------
    Count :
        Class instance implementing all 'count' processes.

    """
    return Count()


class Count:
    """
    Class instance implementing all 'count' processes.

    """

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, condition=None, context=None, dimension=0):
        """
        Gives the number of elements in an array that matches the specified condition.
        Remarks:
            - Counts the number of valid elements by default (condition is set to None).
              A valid element is every element for which is_valid returns True.
            - To count all elements in a list set the `condition` parameter to `True`.

        Parameters
        ----------
        data : np.array
            An array.
        condition : obj, optional
            A condition consists of one ore more processes, which in the end return a boolean value.
            It is evaluated against each element in the array. An element is counted only if the condition
            returns `True`. Defaults to count valid elements in an array (see is_valid). Setting this parameter
            to `True` counts all elements in the array. The following arguments are valid:
                - None : Counts all valid elements, i.e. `is_valid` must yield `True`.
                - `True` : Counts all elements in the array along the specified dimension.
                - object : The following parameters are passed to the process:
                    - `x` : The value of the current element being processed.
                    - `context` : Additional data passed by the user.
        context : dict, optional
            Additional data/keyword arguments to be passed to the condition.
        dimension : int, optional
            Defines the dimension along to count the elements (default is 0).

        Returns
        -------
        count: int
            Count of the data.

        Notes
        -----
        The condition/expression must be able to deal with NumPy arrays.

        """
        if condition is None:
            condition = is_valid
        if condition is True: # explicit check needed
            count = data.shape[dimension]
        elif callable(condition):
            context = context if context is not None else {}
            count = sum(1 for x in data if condition(x, **context))
        else:
            raise ValueError(condition)

        return count

    @staticmethod
    def exec_xar(data, condition=None, context=None, dimension=None):
        """
        Gives the number of elements in an array that matches the specified condition.
        Remarks:
            - Counts the number of valid elements by default (condition is set to None).
              A valid element is every element for which is_valid returns True.
            - To count all elements in a list set the `condition` parameter to `True`.

        Parameters
        ----------
        data : xr.DataArray
            An array.
        condition : obj, optional
            A condition consists of one ore more processes, which in the end return a boolean value.
            It is evaluated against each element in the array. An element is counted only if the condition
            returns `True`. Defaults to count valid elements in an array (see is_valid). Setting this parameter
            to `True` counts all elements in the array. The following arguments are valid:
                - None : Counts all valid elements, i.e. `is_valid` must yield `True`.
                - `True` : Counts all elements in the array along the specified dimension.
                - object : The following parameters are passed to the process:
                    - `x` : The value of the current element being processed.
                    - `context` : Additional data passed by the user.
        context : dict, optional
            Additional data/keyword arguments to be passed to the condition.
        dimension : int, optional
            Defines the dimension along to count the elements (default is 0).

        Returns
        -------
        count: int
            Count of the data.

        Notes
        -----
        The condition/expression must be able to deal with NumPy arrays.
        """
        if condition is None:
            condition = is_valid(data)
        if condition is True:
            count = data.shape[dimension]
        elif callable(condition):
            context = context if context is not None else {}
            if dimension is None:
                count = condition(data, **context).sum()
            elif dimension in data.dims:
                count = condition(data, **context).sum(dim=dimension)
        else:
            raise ValueError(condition)
        return count

    @staticmethod
    def exec_da():
        pass


########################################################################################################################
# Array Apply Process
########################################################################################################################

@process
def array_apply():
    """
    Returns class instance of `ArrayApply`.
    For more details, please have a look at the implementations inside `ArrayApply`.

    Returns
    -------
    ArrayApply :
        Class instance implementing all 'array_apply' processes.

    """
    return ArrayApply()


class ArrayApply:
    """
    Class implementing all 'array_apply' processes.

    """

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, process, context=None):
        """
        Applies a unary process which takes a single value such as `absolute` or `sqrt` to each value in the array.
        This is basically what other languages call either a `for each` loop or a `map` function.

        Parameters
        ----------
        data : np.array
            An array.
        process : callable
            A process to be applied on each value, may consist of multiple sub-processes.
            The specified process must be unary meaning that it must work on a single value.
            The following parameters are passed to the process:
                - `x` : The value of the current element being processed.
                - `context` : Additional data passed by the user.
        context : dict, optional
            Additional data/keyword arguments to be passed to the process.

        Returns
        -------
        np.array :
            An array with the newly computed values. The number of elements are the same as for the original array.

        Notes
        -----
        - The process must be able to deal with NumPy arrays.
        - additional arguments `index` and `label` are ignored as process arguments

        """

        context = context if context is not None else {}
        return process(data, **context)

    @staticmethod
    def exec_xar(data, process, context=None):
        """
        Applies a unary process which takes a single value such as `absolute` or `sqrt` to each value in the array.
        This is basically what other languages call either a `for each` loop or a `map` function.

        Parameters
        ----------
        data : np.array
            An array.
        process : callable
            A process to be applied on each value, may consist of multiple sub-processes.
            The specified process must be unary meaning that it must work on a single value.
            The following parameters are passed to the process:
                - `x` : The value of the current element being processed.
                - `context` : Additional data passed by the user.
        context : dict, optional
            Additional data/keyword arguments to be passed to the process.

        Returns
        -------
        np.array :
            An array with the newly computed values. The number of elements are the same as for the original array.

        Notes
        -----
        - The process must be able to deal with NumPy arrays.
        - additional arguments `index` and `label` are ignored as process arguments
        """
        context = context if context is not None else {}
        return process(data, **context)

    @staticmethod
    def exec_da():
        pass


########################################################################################################################
# Array Filter Process
########################################################################################################################

@process
def array_filter():
    """
    Returns class instance of `ArrayFilter`.
    For more details, please have a look at the implementations inside `ArrayFilter`.

    Returns
    -------
    ArrayFilter :
        Class instance implementing all 'array_filter' processes.

    """
    return ArrayFilter()


class ArrayFilter:
    """
    Class implementing all 'array_filter' processes.

    """

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, condition, context=None):
        """
        Filters the array elements based on a logical expression so that afterwards an array is returned that only
        contains the values conforming to the condition.

        Parameters
        ----------
        data : np.array
            An array.
        condition : callable
            A condition that is evaluated against each value in the array. Only the array elements where the
            condition returns `True` are preserved.
            The following parameters are passed to the process:
                - `x` : The value of the current element being processed.
                - `context` : Additional data passed by the user.
        context : dict, optional
            Additional data/keyword arguments to be passed to the condition.

        Returns
        -------
        np.array :
            An array filtered by the specified condition. The number of elements are less than or equal compared to
            the original array.

        Notes
        -----
        - The condition must be able to deal with NumPy arrays.
        - additional arguments `index` and `label` are ignored as condition arguments

        """

        context = context if context is not None else {}
        return data[condition(data, **context)]

    @staticmethod
    def exec_xar(data, condition, context=None):
        """
        Filters the array elements based on a logical expression so that afterwards an array is returned that only
        contains the values conforming to the condition.

        Parameters
        ----------
        data : xr.DataArray
            An array.
        condition : callable
            A condition that is evaluated against each value in the array. Only the array elements where the
            condition returns `True` are preserved.
            The following parameters are passed to the process:
                - `x` : The value of the current element being processed.
                - `context` : Additional data passed by the user.
        context : dict, optional
            Additional data/keyword arguments to be passed to the condition.

        Returns
        -------
        xr.DataArray :
            An array filtered by the specified condition. The number of elements are less than or equal compared to
            the original array.

        Notes
        -----
        - The condition must be able to deal with NumPy arrays.
        - additional arguments `index` and `label` are ignored as condition arguments

        """
        context = context if context is not None else {}
        data = data.where(condition(data, **context), drop = True)
        data = data.dropna(data.dims[0])
        return data

    @staticmethod
    def exec_da():
        pass


########################################################################################################################
# Array Find Process
########################################################################################################################

@process
def array_find():
    """
    Returns class instance of `ArrayFind`.
    For more details, please have a look at the implementations inside `ArrayFind`.

    Returns
    -------
    ArrayFind :
        Class instance implementing all 'array_find' processes.

    """
    return ArrayFind()


class ArrayFind:
    """
    Class implementing all 'array_find' processes.

    """

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, value, dimension=0):
        """
        Checks whether the array specified for `data` contains the value specified in `value` and returns the
        zero-based index for the first match. If there's no match, np.nan is returned..
        Remarks:
            - All definitions for the process `eq` regarding the comparison of values apply here as well.
              A np.nan return value from eq is handled exactly as `False` (no match).
            - Temporal strings are treated as normal strings and are not interpreted.
            - If the specified value is np.nan, the process always returns np.nan.


        Parameters
        ----------
        data : np.array
            An array to find the value in.
        value : object
            Value to find in `data`.
        dimension : int, optional
            Defines the dimension along to find the value (default is 0).

        Returns
        -------
        int :
            Returns the index of the first element with the specified value.
            If no element was found, np.nan is returned.

        Notes
        -----
        Own implementation, since np.argwhere does not fulfil the requirements.

        """
        if np.isnan(value) or is_empty(data):
            return np.nan
        else:
            bool_idxs = (data == value)
            idxs = np.argmax(bool_idxs, axis=dimension)
            return idxs

    @staticmethod
    def exec_xar(data, value, dimension=None):
        """
        Checks whether the array specified for `data` contains the value specified in `value` and returns the
        zero-based index for the first match. If there's no match, np.nan is returned..
        Remarks:
            - All definitions for the process `eq` regarding the comparison of values apply here as well.
              A np.nan return value from eq is handled exactly as `False` (no match).
            - Temporal strings are treated as normal strings and are not interpreted.
            - If the specified value is np.nan, the process always returns np.nan.


        Parameters
        ----------
        data : xr.DataArray
            An array to find the value in.
        value : object
            Value to find in `data`.
        dimension : str, optional
            Defines the dimension along to find the value (default is None).

        Returns
        -------
        int :
            Returns the index of the first element with the specified value.
            If no element was found, np.nan is returned.

        Notes
        -----
        Own implementation, since np.argmax does not treat 'no matches' right.
        """
        if not dimension:
            dimension = data.dims[0]
        data = data.where(data == value, value-1)
        find = (data.argmax(dim=dimension))
        find_in = data.isin(value).sum(dimension)
        find = find.where(find_in > 0, np.nan)
        return find

    @staticmethod
    def exec_da():
        pass


########################################################################################################################
# Array Labels Process
########################################################################################################################

@process
def array_labels():
    """
    Returns class instance of `ArrayLabels`.
    For more details, please have a look at the implementations inside `ArrayLabels`.

    Returns
    -------
    ArrayLabels :
        Class instance implementing all 'array_labels' processes.

    """
    return ArrayLabels()


class ArrayLabels:
    """
    Class implementing all 'array_labels' processes.

    """

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, dimension=0):
        """
        Returns all labels for a labeled array in the data cube. The labels have the same order as in the array.

        Parameters
        ----------
        data : np.array
            An array with labels.
        dimension : int, optional
            Defines the dimension along to find the labels of the array (default is 0).

        Returns
        -------
        np.array :
            The labels as an array.

        """
        n_vals = data.shape[dimension]
        return np.arange(n_vals)

    @staticmethod
    def exec_xar(data, dimension=None):
        """
        Returns all labels for a labeled array in the data cube. The labels have the same order as in the array.

        Parameters
        ----------
        data : xr.DataArray
            An array with labels.
        dimension : int, str, optional
            Defines the dimension along to find the labels of the array (default is 0).

        Returns
        -------
        xr.DataArray :
            The labels as an array.
        """
        if dimension is None:
            dim = 0
        elif type(dimension) == str:
            i = 0
            while data.dims[i] != dimension and i < len(data.dims):
                i += 1
            dim = i
        else:
            dim = dimension
        n_vals = data.shape[dim]
        return xr.DataArray(np.arange(n_vals))

    @staticmethod
    def exec_da():
        pass


########################################################################################################################
# First Process
########################################################################################################################

@process
def first():
    """
    Returns class instance of `First`.
    For more details, please have a look at the implementations inside `First`.

    Returns
    -------
    First :
        Class instance implementing all 'first' processes.

    """
    return First()


class First:
    """
    Class implementing all 'first' processes.

    """

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0):
        """
        Gives the first element of an array. For an empty array np.nan is returned.

        Parameters
        ----------
        data : np.array
            An array. An empty array resolves always with np.nan.
        ignore_nodata : bool, optional
            Indicates whether no-data values are ignored or not. Ignores them by default (=True).
            Setting this flag to False considers no-data values so that np.nan is returned if any value is such a value.
        dimension : int, optional
            Defines the dimension to select the first element along (default is 0).

        Returns
        -------
        np.array :
            The first element of the input array.

        """
        if is_empty(data):
            return np.nan

        n_dims = len(data.shape)
        if ignore_nodata:  # skip np.nan values
            nan_mask = ~pd.isnull(data)  # create mask for valid values (not np.nan)
            idx_first = np.argmax(nan_mask, axis=dimension)
            first_elem = np.take_along_axis(data, np.expand_dims(idx_first, axis=dimension), axis=dimension)
        else:  # take the first element, no matter np.nan values are in the array
            idx_first = create_slices(0, axis=dimension, n_axes=n_dims)
            first_elem = data[idx_first]

        return first_elem

    @staticmethod
    def exec_xar(data, ignore_nodata=True, dimension=None):
        """
        Gives the first element of an array. For an empty array np.nan is returned.

        Parameters
        ----------
        data : xr.DataArray
            An array. An empty array resolves always with np.nan.
        ignore_nodata : bool, optional
            Indicates whether no-data values are ignored or not. Ignores them by default (=True).
            Setting this flag to False considers no-data values so that np.nan is returned if any value is such a value.
        dimension : int, optional
            Defines the dimension to select the first element along (default is 0).

        Returns
        -------
        xr.DataArray :
            The first element of the input array.
        """
        if len(data) == 0:  # is_empty(data):
            return np.nan
        if dimension is None:
            dimension = 0
        if type(dimension) == str:
            dimension = dimension
        else:
            dimension = data.dims[dimension]
        data_dim = data.transpose(dimension, ...)
        data_first = data_dim[0]
        if ignore_nodata:
            i = 0
            while (data_first.fillna(999) != data_first).any() and i < len(data_dim):
                data_first = data_first.fillna(data_dim[i])
                i += 1
        return data_first

    @staticmethod
    def exec_da():
        pass


########################################################################################################################
# Last Process
########################################################################################################################

@process
def last():
    """
    Returns class instance of `Last`.
    For more details, please have a look at the implementations inside `Last`.

    Returns
    -------
    Last :
        Class instance implementing all 'last' processes.

    """
    return Last()


class Last:
    """
    Class implementing all 'last' processes.

    """

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, ignore_nodata=True, dimension=0):
        """
        Gives the last element of an array. For an empty array np.nan is returned.

        Parameters
        ----------
        data : np.array
            An array. An empty array resolves always with np.nan.
        ignore_nodata : bool, optional
            Indicates whether no-data values are ignored or not. Ignores them by default (=True).
            Setting this flag to False considers no-data values so that np.nan is returned if any value is such a value.
        dimension : int, optional
            Defines the dimension to select the last element along (default is 0).

        Returns
        -------
        np.array :
            The last element of the input array.

        """
        if is_empty(data):
            return np.nan

        n_dims = len(data.shape)
        if ignore_nodata:  # skip np.nan values
            data = np.flip(data, axis=dimension)  # flip data to retrieve the first valid element (thats the only way it works with argmax)
            last_elem = first(data, ignore_nodata=ignore_nodata, dimension=dimension)
        else:  # take the first element, no matter np.nan values are in the array
            idx_last = create_slices(-1, axis=dimension, n_axes=n_dims)
            last_elem = data[idx_last]

        return last_elem

    @staticmethod
    def exec_xar(data, ignore_nodata=True, dimension=None):
        """
        Gives the last element of an array. For an empty array np.nan is returned.

        Parameters
        ----------
        data : xr.DataArray
            An array. An empty array resolves always with np.nan.
        ignore_nodata : bool, optional
            Indicates whether no-data values are ignored or not. Ignores them by default (=True).
            Setting this flag to False considers no-data values so that np.nan is returned if any value is such a value.
        dimension : int, optional
            Defines the dimension to select the last element along (default is 0).

        Returns
        -------
        xr.DataArray :
            The last element of the input array.
        """
        if len(data) == 0:  # is_empty(data):
            return np.nan
        if dimension is None:
            dimension = 0
        if type(dimension) == str:
            dimension = dimension
        else:
            dimension = data.dims[dimension]
        data_dim = data.transpose(dimension, ...)
        data_last = data_dim[-1]
        if ignore_nodata:
            i = len(data_dim) - 1
            while (data_last.fillna(999) != data_last).any() and i >= 0:
                data_last = data_last.fillna(data_dim[i])
                i -= 1
        return data_last

    @staticmethod
    def exec_da():
        pass


########################################################################################################################
# Order Process
########################################################################################################################

@process
def order():
    """
    Returns class instance of `Order`.
    For more details, please have a look at the implementations inside `Order`.

    Returns
    -------
    Order :
        Class instance implementing all 'order' processes.

    """
    return Order()


# TODO: can nodata algorithm be simplified/enhanced?
class Order:
    """
    Class implementing all 'order' processes.

    """

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, asc=True, nodata=None, dimension=0):
        """
        Computes a permutation which allows rearranging the data into ascending or descending order.
        In other words, this process computes the ranked (sorted) element positions in the original list.
        Remarks:
            - The positions in the result are zero-based.
            - Ties will be left in their original ordering.

        Parameters
        ----------
        data : np.array
            An array to compute the order for.
        dimension : int, optional
            Defines the dimension to order along (default is 0).
        asc : bool, optional
            The default sort order is ascending, with smallest values first. To sort in reverse (descending) order,
            set this parameter to `False`.
        nodata : obj, optional
            Controls the handling of no-data values (np.nan). By default they are removed. If `True`, missing values
            in the data are put last; if `False`, they are put first.

        Returns
        -------
        np.array :
            The computed permutation.

        Notes
        -----
        - the case with nodata=False is complicated, since a simple nan masking destroys the structure of the array
        - due to the flipping, the order of the np.nan values is wrong, but this is ignored, since this order should
          not be relevant
        """

        if asc:
            permutation_idxs = np.argsort(data, kind='mergesort', axis=dimension)
        else:  # [::-1] not possible
            permutation_idxs = np.argsort(-data, kind='mergesort', axis=dimension)  # to get the indizes in descending order, the sign of the data is changed

        if nodata is None:  # ignore np.nan values
            # sort the original data first, to get correct position of no data values
            sorted_data = data[permutation_idxs]
            return permutation_idxs[~pd.isnull(sorted_data)]
        elif nodata is False:  # put location/index of np.nan values first
            # sort the original data first, to get correct position of no data values
            sorted_data = data[permutation_idxs]
            nan_idxs = pd.isnull(sorted_data)

            # flip permutation and nan mask
            permutation_idxs_flip = np.flip(permutation_idxs, axis=dimension)
            nan_idxs_flip = np.flip(nan_idxs, axis=dimension)

            # flip causes the nan.values to be first, however the order of all other values is also flipped
            # therefore the non np.nan values (i.e. the wrong flipped order) is replaced by the right order given by
            # the original permutation values
            permutation_idxs_flip[~nan_idxs_flip] = permutation_idxs[~nan_idxs]

            return permutation_idxs_flip
        elif nodata is True:  # default argsort behaviour, np.nan values are put last
            return permutation_idxs
        else:
            err_msg = "Data type of 'nodata' argument is not supported."
            raise Exception(err_msg)

    @staticmethod
    def exec_xar(data, asc=True, nodata=None, dimension=None):
        """
        Computes a permutation which allows rearranging the data into ascending or descending order.
        In other words, this process computes the ranked (sorted) element positions in the original list.
        Remarks:
            - The positions in the result are zero-based.
            - Ties will be left in their original ordering.

        Parameters
        ----------
        data : xr.DataArray
            An array to compute the order for.
        dimension : int, str, optional
            Defines the dimension to order along (default is 0).
        asc : bool, optional
            The default sort order is ascending, with smallest values first. To sort in reverse (descending) order,
            set this parameter to `False`.
        nodata : obj, optional
            Controls the handling of no-data values (np.nan). By default they are removed. If `True`, missing values
            in the data are put last; if `False`, they are put first.

        Returns
        -------
        xr.DataArray :
            The computed permutation.

        Notes
        -----
        - the case with nodata=False is complicated, since a simple nan masking destroys the structure of the array
        - due to the flipping, the order of the np.nan values is wrong, but this is ignored, since this order should
          not be relevant
        """
        if len(data) == 0:
            return np.nan
        if dimension is None:
            dimension = 0
        if type(dimension) == str:
            dimension_str = dimension
        else:
            dimension_str = data.dims[dimension]
        if nodata is None:
            data = data.dropna(dimension_str)
        k = len(data[dimension_str].values)
        if (asc and not nodata) or (not asc and nodata):
            fill = data.min() - 1
            data = data.fillna(fill)
        order = argtopk(data, k = k, dim = dimension_str)
        if asc:
            r = order[dimension_str].values
            r = np.flip(r)
            order = order.loc[{dimension_str: r}]
        order = order.transpose(*data.dims)
        return order

    @staticmethod
    def exec_da():
        pass


########################################################################################################################
# Rearrange Process
########################################################################################################################

@process
def rearrange():
    """
    Returns class instance of `Rearrange`.
    For more details, please have a look at the implementations inside `Rearrange`.

    Returns
    -------
    Rearrange :
        Class instance implementing all 'rearrange' processes.

    """
    return Rearrange()


class Rearrange:
    """
    Class implementing all 'rearrange' processes.

    """

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, order):
        """
        Rearranges an array based on a permutation, i.e. a ranked list of element positions in the original list.
        The positions must be zero-based.

        Parameters
        ----------
        data : np.array
            The array to rearrange.
        order : np.array
            The permutation used for rearranging.

        Returns
        -------
        np.array :
            The rearranged array.

        """

        return data[order]

    @staticmethod
    def exec_xar(data, order):
        """
        Rearranges an array based on a permutation, i.e. a ranked list of element positions in the original list.
        The positions must be zero-based.

        Parameters
        ----------
        data : xr.DataArray
            The array to rearrange.
        order : xr.DataArray
            The permutation used for rearranging.

        Returns
        -------
        xr.DataArray :
            The rearranged array.

        """

        return data[order]

    @staticmethod
    def exec_da():
        pass


########################################################################################################################
# Sort Process
########################################################################################################################

@process
def sort():
    """
    Returns class instance of `Sort`.
    For more details, please have a look at the implementations inside `Sort`.

    Returns
    -------
    Sort :
        Class instance implementing all 'sort' processes.

    """
    return Sort()


# TODO: can nodata=False algorithm be simplified?
class Sort:
    """
    Class implementing all 'sort' processes.

    """

    @staticmethod
    def exec_num():
        pass

    @staticmethod
    def exec_np(data, asc=True, nodata=None, dimension=0):
        """
        Sorts an array into ascending (default) or descending order.
        Remarks:
            - Ties will be left in their original ordering.

        Parameters
        ----------
        data : np.array
            An array with data to sort.
        dimension : int, optional
            Defines the dimension to sort along (default is 0).
        asc : bool, optional
            The default sort order is ascending, with smallest values first. To sort in reverse (descending) order,
            set this parameter to `False`.
        nodata : obj, optional
            Controls the handling of no-data values (np.nan). By default they are removed. If `True`, missing values
            in the data are put last; if `False`, they are put first.

        Returns
        -------
        np.array :
            The sorted array.

        """
        if asc:
            data_sorted = np.sort(data, axis=dimension)
        else:  # [::-1] not possible
            data_sorted = -np.sort(-data, axis=dimension)  # to get the indexes in descending order, the sign of the data is changed

        if nodata is None:  # ignore np.nan values
            nan_idxs = pd.isnull(data_sorted)
            return data_sorted[~nan_idxs]
        elif nodata == False:  # put np.nan values first
            nan_idxs = pd.isnull(data_sorted)
            data_sorted_flip = np.flip(data_sorted, axis=dimension)
            nan_idxs_flip = pd.isnull(data_sorted_flip)
            data_sorted_flip[~nan_idxs_flip] = data_sorted[~nan_idxs]
            return data_sorted_flip
        elif nodata == True:  # default sort behaviour, np.nan values are put last
            return data_sorted
        else:
            err_msg = "Data type of 'nodata' argument is not supported."
            raise Exception(err_msg)

    @staticmethod
    def exec_xar(data, asc=True, nodata=None, dimension=None):
        """
        Sorts an array into ascending (default) or descending order.
        Remarks:
            - Ties will be left in their original ordering.

        Parameters
        ----------
        data : xr.DataArray
            An array with data to sort.
        dimension : int, str, optional
            Defines the dimension to sort along (default is 0).
        asc : bool, optional
            The default sort order is ascending, with smallest values first. To sort in reverse (descending) order,
            set this parameter to `False`.
        nodata : obj, optional
            Controls the handling of no-data values (np.nan). By default they are removed. If `True`, missing values
            in the data are put last; if `False`, they are put first.

        Returns
        -------
        xr.DataArray :
            The sorted array.
        """
        if len(data) == 0:
            return np.nan
        if dimension is None:
            dimension = 0
        if type(dimension) == str:
            dimension_str = dimension
        else:
            dimension_str = data.dims[dimension]
        if nodata is None:
            data = data.dropna(dimension_str)
        fill = None
        if asc:
            k = (-1)*len(data[dimension_str].values)
            if not nodata:
                fill = data.min()-1
                data = data.fillna(fill)
        else:
            k = len(data[dimension_str].values)
            if nodata:
                fill = data.min() - 1
                data = data.fillna(fill)
        sorted = topk(data, k = k, dim = dimension_str)
        sorted = sorted.transpose(*data.dims)
        if fill is not None:
            sorted = sorted.where(sorted != fill, np.nan)
        return sorted

    @staticmethod
    def exec_da():
        pass

