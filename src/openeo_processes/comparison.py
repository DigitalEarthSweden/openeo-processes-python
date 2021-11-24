import datetime
import math

import numpy as np
import xarray as xr
import dask as da
from openeo_processes.utils import process
from openeo_processes.utils import str2time
from openeo_processes.utils import keep_attrs


# TODO: test if this works for different data types
def is_empty(data):
    """
    Checks if an object is empty (its length is zero) or not.

    Parameters
    ----------
    data : object
        Any Class instance, which has a __length__ class method.

    Returns
    -------
    bool :
        True if object is emtpy, False if not.

    """
    return len(data) == 0


########################################################################################################################
# Is Nodata Process
########################################################################################################################

@process
def is_nodata():
    """
    Returns class instance of `IsNodata`.
    For more details, please have a look at the implementations inside `IsNodata`.

    Returns
    -------
    IsNodata :
        Class instance implementing all 'is_nodata' processes.

    """
    return IsNodata()


class IsNodata:
    """
    Class implementing all 'is_nodata' processes.

    """

    @staticmethod
    def exec_num(x):
        """
        Checks whether the specified data is a missing data, i.e. equals to a no-data value/None.

        Parameters
        ----------
        x : float or int
            The data to check.

        Returns
        -------
        bool :
            True if the data is a no-data value/None, otherwise False.

        """
        return x is None

    @staticmethod
    def exec_np(x):
        """
        Checks whether the specified data is a missing data, i.e. equals to a no-data value/None.

        Parameters
        ----------
        x : np.array
            The data to check.

        Returns
        -------
        np.array :
            Array with True values if the data is a no-data value/None, otherwise False values.

        Notes
        -----
        Attention! Since None values are not supported NumPy and Pandas, this method has the same behaviour as `is_nan`.

        """
        return IsNodata.exec_num(x)

    @staticmethod
    def exec_xar(x):
        return IsNodata.exec_num(x)

    @staticmethod
    def exec_da(x):
        return IsNodata.exec_num(x)


########################################################################################################################
# Is NaN Process
########################################################################################################################

@process
def is_nan():
    """
    Returns class instance of `IsNan`.
    For more details, please have a look at the implementations inside `IsNan`.

    Returns
    -------
    IsNan :
        Class instance implementing all 'is_nan' processes.

    """
    return IsNan()


class IsNan:
    """
    Class implementing all 'is_nan' processes.

    """

    @staticmethod
    def exec_num(x):
        """
        Checks whether the specified value `x` is not a number (often abbreviated as NaN).
        The definition of NaN follows the IEEE Standard 754. All non-numeric data types also return True.

        Parameters
        ----------
        x : int or float or str
            The data to check.

        Returns
        -------
        bool :
            True if the data is a not a number, otherwise False.

        """
        return (not isinstance(x, (int, float))) or math.isnan(x)

    @staticmethod
    def exec_np(x):
        """
        Checks whether the specified array `x` contains values being not a number (often abbreviated as NaN).
        The definition of NaN follows the IEEE Standard 754. All non-numeric data types also return True.

        Parameters
        ----------
        x : np.array
            The data to check.

        Returns
        -------
        bool :
            True if the data is a not a number, otherwise False.

        """
        return IsNan.exec_num(x)

    @staticmethod
    def exec_xar(x):
        return IsNan.exec_num(x)

    @staticmethod
    def exec_da(x):
        return IsNan.exec_num(x)


########################################################################################################################
# Is Valid Process
########################################################################################################################

@process
def is_valid():
    """
    Returns class instance of `IsValid`.
    For more details, please have a look at the implementations inside `IsValid`.

    Returns
    -------
    IsValid :
        Class instance implementing all 'is_valid' processes.

    """
    return IsValid()


class IsValid:
    """
    Class implementing all 'is_valid' processes.

    """

    @staticmethod
    def exec_num(x):
        """
        Checks whether the specified value `x` is valid. A value is considered valid if it is
            - not a no-data value (null) and
            - a finite number (only if it is a number). The definition of finite and infinite numbers follows the
              IEEE Standard 754.

        Parameters
        ----------
        x : float or int or str
            The data to check.

        Returns
        -------
        bool :
            True if the data is valid, otherwise False.

        """
        if x is None:
            return False
        elif isinstance(x, float):
            return not (math.isnan(x) or math.isinf(x))
        return True

    @staticmethod
    def exec_np(x):
        """
        Checks whether the specified array `x` contains valid values. A value is considered valid if it is
            - not a no-data value (null) and
            - a finite number (only if it is a number). The definition of finite and infinite numbers follows the
              IEEE Standard 754.

        Parameters
        ----------
        x : float or int or str
            The data to check.

        Returns
        -------
        bool :
            True if the data is valid, otherwise False.

        """
        return IsValid.exec_num(x)

    @staticmethod
    def exec_xar(x):
        return IsValid.exec_num(x)

    @staticmethod
    def exec_da(x):
        return IsValid.exec_num(x)


########################################################################################################################
# Eq Process
########################################################################################################################

@process
def eq():
    """
    Returns class instance of `Eq`.
    For more details, please have a look at the implementations inside `Eq`.

    Returns
    -------
    Eq :
        Class instance implementing all 'eq' processes.

    """
    return Eq()


class Eq:
    """
    Class instance implementing all 'eq' processes.

    """

    @staticmethod
    def exec_num(x, y, delta=None, case_sensitive=True):
        """
        Compares whether `x` is strictly equal to `y`.
        Remarks:
            - An integer 1 is equal to a floating point number 1.0 as integer is a sub-type of number.
            - If any operand is None, the return value is None.
              Therefore, exec_num(None, None) returns None instead of True.
            - Strings are expected to be encoded in UTF-8 by default.
            - Temporal strings are differently than other strings and are not be compared based on their string
              representation due to different possible representations. For example, the UTC time zone
              representation Z has the same meaning as +00:00.

        Parameters
        ----------
        x : float or int or str or datetime.datetime
            First operand.
        y : float or int or str or datetime.datetime
            Second operand.
        delta : float, optional
            Only applicable for comparing two numbers. If this optional parameter is set to a positive non-zero number
            the equality of two numbers is checked against a delta value. This is especially useful to circumvent
            problems with floating point inaccuracy in machine-based computation.
        case_sensitive : bool, optional
            Only applicable for comparing two strings. Case sensitive comparison can be disabled by setting this
            parameter to False.

        Returns
        -------
        bool :
            Returns True if `x` is equal to `y`, None if any operand is None, otherwise False.

        """
        if x is None or y is None:
            return None

        if (type(x) in [float, int]) and (type(y) in [float, int]):  # comparison of numbers
            if type(delta) in [float, int]:
                return np.isclose(x, y, atol=delta)
            else:
                return x == y
        elif (type(x) == str) and (type(y) == str):  # comparison of strings or dates
            # try to convert the string into a date
            x_time = str2time(x)
            y_time = str2time(y)
            if x_time is None or y_time is None: # comparison of strings
                if case_sensitive:
                    return x == y
                else:
                    return x.lower() == y.lower()
            else:
                return x_time == y_time  # comparison of dates
        else:
            return False

    @staticmethod
    def exec_np(x, y, delta=None, case_sensitive=True, reduce=False):  # TODO: add equal checks for date strings
        """
        Compares whether `x` is strictly equal to `y`.

        Parameters
        ----------
        x : np.ndarray
            First operand.
        y : np.ndarray
            Second operand.
        delta : float, optional
            Only applicable for comparing two arrays containing numbers. If this optional parameter is set to a
            positive non-zero number the equality of two numbers is checked against a delta value. This is especially
            useful to circumvent problems with floating point inaccuracy in machine-based computation.
        case_sensitive : bool, optional
            Only applicable for comparing two string arrays. Case sensitive comparison can be disabled by setting this
            parameter to False.
        reduce : bool, optional
            If True, one value will be returned, i.e. if the arrays are equal.
            If False, each value in `x` will be compared with the respective value in `y`. Defaults to False.

        Returns
        -------
        bool or np.ndarray :
            Returns True if `x` is equal to `y`, np.nan if any operand is np.nan, otherwise False.

        """
        if x is None or y is None:
            return None

        if x.dtype.kind.lower() in ['f', 'i'] and y.dtype.kind.lower() in ['f', 'i']:  # both arrays only contain numbers
            if type(delta) in [float, int]:
                ar_eq = np.isclose(x, y, atol=delta)
            else:
                ar_eq = x == y
        else:
            ar_eq = x == y

        if reduce:
            return ar_eq.all()
        else:
            return ar_eq

    @staticmethod
    def exec_xar(x, y, delta=False, case_sensitive=True, reduce=False): # TODO: add equal checks for date strings in xar
        """
        Compares whether `x` is strictly equal to `y`.

        Parameters
        ----------
        x : xr.DataArray, integer, float
            First operand.
        y : xr.DataArray, integer, float
            Second operand.
        delta : float, optional
            Only applicable for comparing two arrays containing numbers. If this optional parameter is set to a
            positive non-zero number the equality of two numbers is checked against a delta value. This is especially
            useful to circumvent problems with floating point inaccuracy in machine-based computation.
        case_sensitive : bool, optional
            Only applicable for comparing two string arrays. Case sensitive comparison can be disabled by setting this
            parameter to False.
        reduce : bool, optional
            If True, one value will be returned, i.e. if the arrays are equal.
            If False, each value in `x` will be compared with the respective value in `y`. Defaults to False.

        Returns
        -------
        bool or xr.DataArray :
            Returns True if `x` is equal to `y`, np.nan if any operand is np.nan, otherwise False.

        """
        if x is None or y is None:
            return None

        x_type = x.dtype if isinstance(x, (xr.core.dataarray.DataArray, np.ndarray)) else type(x)
        y_type = y.dtype if isinstance(y, (xr.core.dataarray.DataArray, np.ndarray)) else type(y)

        if (x_type in [float, int]) and (y_type in [float, int]):  # both arrays only contain numbers
            if type(delta) in [float, int]:
                ar_eq = (abs(x-y) <= delta)
            else:
                ar_eq = x == y
        elif (isinstance(x, str) or x_type.kind.lower() == 'u') and (isinstance(y, str) or y_type.kind.lower() == 'u'):  # comparison of strings or dates
            # try to convert the string into a date
            x_time = None # str2time still missing
            y_time = None
            if x_time is None or y_time is None: # comparison of strings
                if case_sensitive:
                    ar_eq = x == y
                else:
                    x_l = x.str.lower() if isinstance(x, (xr.core.dataarray.DataArray, np.ndarray)) else x.lower()
                    y_l = y.str.lower() if isinstance(y, (xr.core.dataarray.DataArray, np.ndarray)) else y.lower()
                    ar_eq = x_l == y_l
            else:
                ar_eq = x_time == y_time  # comparison of dates
        else:
            ar_eq = x == y
        ar_eq = keep_attrs(x, y, ar_eq)
        if reduce:
            return ar_eq.all()
        else:
            return ar_eq

    @staticmethod
    def exec_da():
        pass


########################################################################################################################
# Neq Process
########################################################################################################################

@process
def neq():
    """
    Returns class instance of `Neq`.
    For more details, please have a look at the implementations inside `Neq`.

    Returns
    -------
    Neq :
        Class instance implementing all 'neq' processes.

    """
    return Neq()


class Neq:
    """
    Class instance implementing all 'neq' processes.

    """

    @staticmethod
    def exec_num(x, y, delta=None, case_sensitive=True):
        """
        Compares whether `x` is not strictly equal to `y`.
        Remarks:
            - An integer 1 is equal to a floating point number 1.0 as integer is a sub-type of number.
            - If any operand is None, the return value is None.
              Therefore, exec_num(None, None) returns None instead of False.
            - Strings are expected to be encoded in UTF-8 by default.
            - Temporal strings are differently than other strings and are not be compared based on their string
              representation due to different possible representations. For example, the UTC time zone
              representation Z has the same meaning as +00:00.

        Parameters
        ----------
        x : float or int or str or datetime.datetime
            First operand.
        y : float or int or str or datetime.datetime
            Second operand.
        delta : float, optional
            Only applicable for comparing two numbers. If this optional parameter is set to a positive non-zero number
            the non-equality of two numbers is checked against a delta value. This is especially useful to circumvent
            problems with floating point inaccuracy in machine-based computation.
        case_sensitive : bool, optional
            Only applicable for comparing two strings. Case sensitive comparison can be disabled by setting this
            parameter to False.

        Returns
        -------
        bool :
            Returns True if `x` is not equal to `y`, None if any operand is None, otherwise False.

        """
        eq_val = Eq().exec_num(x, y, delta=delta, case_sensitive=case_sensitive)
        if eq_val is None:
            return None
        else:
            return not eq_val

    @staticmethod
    def exec_np(x, y, delta=None, case_sensitive=True, reduce=False):  # TODO: add equal checks for date strings
        """
        Compares whether `x` is strictly equal to `y`.

        Parameters
        ----------
        x : np.ndarray
            First operand.
        y : np.ndarray
            Second operand.
        delta : float, optional
            Only applicable for comparing two arrays containing numbers. If this optional parameter is set to a
            positive non-zero number the equality of two numbers is checked against a delta value. This is especially
            useful to circumvent problems with floating point inaccuracy in machine-based computation.
        case_sensitive : bool, optional
            Only applicable for comparing two string arrays. Case sensitive comparison can be disabled by setting this
            parameter to False.
        reduce : bool, optional
            If True, one value will be returned, i.e. if the arrays are equal.
            If False, each value in `x` will be compared with the respective value in `y`. Defaults to False.

        Returns
        -------
        bool or np.ndarray :
            Returns True if `x` is not equal to `y`, np.nan if any operand is np.nan, otherwise False.

        """
        eq_val = Eq().exec_np(x, y, delta=delta, case_sensitive=case_sensitive, reduce=reduce)
        if eq_val is None:
            return None
        else:
            return ~eq_val

    @staticmethod
    def exec_xar(x, y, delta=None, case_sensitive=True, reduce=False):  # TODO: add equal checks for date strings
        """
        Compares whether `x` is strictly equal to `y`.

        Parameters
        ----------
        x : xr.DataArray, integer, float
            First operand.
        y : xr.DataArray, integer, float
            Second operand.
        delta : float, optional
            Only applicable for comparing two arrays containing numbers. If this optional parameter is set to a
            positive non-zero number the equality of two numbers is checked against a delta value. This is especially
            useful to circumvent problems with floating point inaccuracy in machine-based computation.
        case_sensitive : bool, optional
            Only applicable for comparing two string arrays. Case sensitive comparison can be disabled by setting this
            parameter to False.
        reduce : bool, optional
            If True, one value will be returned, i.e. if the arrays are equal.
            If False, each value in `x` will be compared with the respective value in `y`. Defaults to False.

        Returns
        -------
        bool or xr.DataArray :
            Returns True if `x` is not equal to `y`, np.nan if any operand is np.nan, otherwise False.

        """
        eq_val = Eq().exec_xar(x, y, delta=delta, case_sensitive=case_sensitive, reduce=reduce)
        if eq_val is None:
            return None
        else:
            return da.array.logical_not(eq_val)

    @staticmethod
    def exec_da():
        pass


########################################################################################################################
# Gt Process
########################################################################################################################

@process
def gt():
    """
    Returns class instance of `Gt`.
    For more details, please have a look at the implementations inside `Gt`.

    Returns
    -------
    Gt :
        Class instance implementing all 'gt' processes.

    """
    return Gt()


class Gt:
    """
    Class instance implementing all 'gt' processes.

    """

    @staticmethod
    def exec_num(x, y):
        """
        Compares whether `x` is strictly greater than `y`.
        Remarks:
            - If any operand is None, the return value is None.
            - If any operand is not a number or temporal string (date, time or date-time), the process returns False.
            - Temporal strings can not be compared based on their string representation due to the time zone /
            time-offset representations.

        Parameters
        ----------
        x : float or int or datetime.datetime
            First operand.
        y : float or int or datetime.datetime
            Second operand.

        Returns
        -------
        bool :
            Returns True if `x` is strictly greater than `y`, None if any operand is None, otherwise False.

        """
        if x is None or y is None:
            return None
        elif isinstance(x, str) and isinstance(y, str):
            return str2time(x) > str2time(y)
        elif isinstance(x, (int, float, datetime.datetime)) and isinstance(y, (int, float, datetime.datetime)):
            return x > y
        else:
            return False

    @staticmethod
    def exec_np(x, y, reduce=False):
        """
        Compares whether `x` is strictly greater than `y`.
        Remarks:
            - If any operand is None, the return value is None.
            - If any operand is not a number or temporal string (date, time or date-time), the process returns False.
            - Temporal strings can not be compared based on their string representation due to the time zone /
            time-offset representations.

        Parameters
        ----------
        x : np.ndarray
            First operand.
        y : np.ndarray
            Second operand.
        reduce : bool, optional
            If True, one value will be returned.
            If False, each value in `x` will be compared with the respective value in `y`. Defaults to False.

        Returns
        -------
        bool :
            Returns True if `x` is strictly greater than `y`, None if any operand is None, otherwise False.

        """
        if x is None or y is None:
            return None
        elif x.dtype.kind.lower() in ['f', 'i', 'm']:
            gt_ar = x > y
            if reduce:
                return gt_ar.all()
            else:
                return gt_ar
        else:
            if reduce:
                return False
            else:
                return np.zeros(x.shape, dtype=np.bool)

    @staticmethod
    def exec_xar(x, y, reduce=False):
        """
        Compares whether `x` is strictly greater than `y`.
        Remarks:
            - If any operand is None, the return value is None.
            - If any operand is not a number or temporal string (date, time or date-time), the process returns False.
            - Temporal strings can not be compared based on their string representation due to the time zone /
            time-offset representations.

        Parameters
        ----------
        x : xr.DataArray, integer, float
            First operand.
        y : xr.DataArray, integer, float
            Second operand.
        reduce : bool, optional
            If True, one value will be returned.
            If False, each value in `x` will be compared with the respective value in `y`. Defaults to False.

        Returns
        -------
        xr.DataArray: :
            Returns True if `x` is strictly greater than `y`, None if any operand is None, otherwise False.
        """

        if x is None or y is None:
            return None
        elif type(x) in [int, float, xr.DataArray] and type(y) in [int, float, xr.DataArray]:
            gt_ar = x > y
            gt_ar = keep_attrs(x, y, gt_ar)
            if reduce:
                return gt_ar.all()
            else:
                return gt_ar
        return False

    @staticmethod
    def exec_da():
        pass


########################################################################################################################
# Gte Process
########################################################################################################################

@process
def gte():
    """
    Returns class instance of `Gte`.
    For more details, please have a look at the implementations inside `Gte`.

    Returns
    -------
    Gte :
        Class instance implementing all 'gte' processes.

    """
    return Gte()


class Gte:
    """
    Class instance implementing all 'gte' processes.

    """

    @staticmethod
    def exec_num(x, y):
        """
        Compares whether `x` is strictly greater than or equal to `y`.
        Remarks:
            - If any operand is None, the return value is None.
            - If any operand is not a number or temporal string (date, time or date-time), the process returns False.
            - Temporal strings can not be compared based on their string representation due to the time zone /
            time-offset representations.

        Parameters
        ----------
        x : float or int or datetime.datetime
            First operand.
        y : float or int or datetime.datetime
            Second operand.

        Returns
        -------
        bool :
            Returns True if `x` is strictly greater than or equal to `y`, None if any operand is None, otherwise False.

        """
        if x is None or y is None:
            return None
        elif isinstance(x, str) and isinstance(y, str):
            return str2time(x) >= str2time(y)
        elif isinstance(x, (int, float, datetime.datetime)) and isinstance(y, (int, float, datetime.datetime)):
            return x >= y
        else:
            return False

    @staticmethod
    def exec_np(x, y, reduce=False):
        """
        Compares whether `x` is strictly greater than or equal to `y`.
        Remarks:
            - If any operand is None, the return value is None.
            - If any operand is not a number or temporal string (date, time or date-time), the process returns False.
            - Temporal strings can not be compared based on their string representation due to the time zone /
            time-offset representations.

        Parameters
        ----------
        x : np.ndarray
            First operand.
        y : np.ndarray
            Second operand.
        reduce : bool, optional
            If True, one value will be returned.
            If False, each value in `x` will be compared with the respective value in `y`. Defaults to False.

        Returns
        -------
        bool :
            Returns True if `x` is strictly greater than or equal to `y`, None if any operand is None, otherwise False.

        """
        if x is None or y is None:
            return None
        elif x.dtype.kind.lower() in ['f', 'i', 'm']:
            gte_ar = x >= y
            if reduce:
                return gte_ar.all()
            else:
                return gte_ar
        else:
            if reduce:
                return False
            else:
                return np.zeros(x.shape, dtype=np.bool)

    @staticmethod
    def exec_xar(x, y, reduce = False):
        """
        Compares whether `x` is strictly greater than or equal to `y`.
        Remarks:
            - If any operand is None, the return value is None.
            - If any operand is not a number or temporal string (date, time or date-time), the process returns False.
            - Temporal strings can not be compared based on their string representation due to the time zone /
            time-offset representations.

        Parameters
        ----------
        x : xr.DataArray, integer, float
            First operand.
        y : xr.DataArray, integer, float
            Second operand.
        reduce : bool, optional
            If True, one value will be returned.
            If False, each value in `x` will be compared with the respective value in `y`. Defaults to False.

        Returns
        -------
        bool :
            Returns True if `x` is strictly greater than or equal to `y`, None if any operand is None, otherwise False.

        """
        if x is None or y is None:
            return None
        elif type(x) in [int, float, xr.DataArray] and type(y) in [int, float, xr.DataArray]:
            gte_ar = ((x-y) >= 0)
            gte_ar = keep_attrs(x, y, gte_ar)
            if reduce:
                return gte_ar.all()
            else:
                return gte_ar
        return False

    @staticmethod
    def exec_da():
        pass


########################################################################################################################
# Lt Process
########################################################################################################################

@process
def lt():
    """
    Returns class instance of `Lt`.
    For more details, please have a look at the implementations inside `Lt`.

    Returns
    -------
    Lt :
        Class instance implementing all 'lt' processes.

    """
    return Lt()


class Lt:
    """
    Class instance implementing all 'lt' processes.

    """

    @staticmethod
    def exec_num(x, y):
        """
        Compares whether `x` is strictly lower than `y`.
        Remarks:
            - If any operand is None, the return value is None.
            - If any operand is not a number or temporal string (date, time or date-time), the process returns False.
            - Temporal strings can not be compared based on their string representation due to the time zone /
            time-offset representations.

        Parameters
        ----------
        x : float or int or datetime.datetime
            First operand.
        y : float or int or datetime.datetime
            Second operand.

        Returns
        -------
        bool :
            Returns True if `x` is strictly lower than `y`, None if any operand is None, otherwise False.

        """
        if x is None or y is None:
            return None
        elif isinstance(x, str) and isinstance(y, str):
            return str2time(x) < str2time(y)
        elif isinstance(x, (int, float, datetime.datetime)) and isinstance(y, (int, float, datetime.datetime)):
            return x < y
        else:
            return False

    @staticmethod
    def exec_np(x, y, reduce=False):
        """
        Compares whether `x` is strictly lower than `y`.
        Remarks:
            - If any operand is None, the return value is None.
            - If any operand is not a number or temporal string (date, time or date-time), the process returns False.
            - Temporal strings can not be compared based on their string representation due to the time zone /
            time-offset representations.

        Parameters
        ----------
        x : np.ndarray
            First operand.
        y : np.ndarray
            Second operand.
        reduce : bool, optional
            If True, one value will be returned.
            If False, each value in `x` will be compared with the respective value in `y`. Defaults to False.

        Returns
        -------
        bool :
            Returns True if `x` is strictly lower than `y`, None if any operand is None, otherwise False.

        """
        if x is None or y is None:
            return None
        elif x.dtype.kind.lower() in ['f', 'i', 'm']:
            lt_ar = x < y
            if reduce:
                return lt_ar.all()
            else:
                return lt_ar
        else:
            if reduce:
                return False
            else:
                return np.zeros(x.shape, dtype=np.bool)

    @staticmethod
    def exec_xar(x, y, reduce = False):
        """
        Compares whether `x` is strictly lower than `y`.
        Remarks:
            - If any operand is None, the return value is None.
            - If any operand is not a number or temporal string (date, time or date-time), the process returns False.
            - Temporal strings can not be compared based on their string representation due to the time zone /
            time-offset representations.

        Parameters
        ----------
        x : xr.DataArray, integer, float
            First operand.
        y : xr.DataArray, integer, float
            Second operand.
        reduce : bool, optional
            If True, one value will be returned.
            If False, each value in `x` will be compared with the respective value in `y`. Defaults to False.

        Returns
        -------
        bool :
            Returns True if `x` is strictly lower than `y`, None if any operand is None, otherwise False.

        """
        if x is None or y is None:
            return None
        elif type(x) in [int, float, xr.DataArray] and type(y) in [int, float, xr.DataArray]:
            lt_ar = x < y
            lt_ar = keep_attrs(x, y, lt_ar)
            if reduce:
                return lt_ar.all()
            else:
                return lt_ar
        return False

    @staticmethod
    def exec_da():
        pass


########################################################################################################################
# Lte Process
########################################################################################################################

@process
def lte():
    """
    Returns class instance of `Lte`.
    For more details, please have a look at the implementations inside `Lte`.

    Returns
    -------
    Lte :
        Class instance implementing all 'lte' processes.

    """
    return Lte()


class Lte:
    """
    Class instance implementing all 'lte' processes.

    """

    @staticmethod
    def exec_num(x, y):
        """
        Compares whether `x` is strictly lower than or equal to `y`.
        Remarks:
            - If any operand is None, the return value is None.
            - If any operand is not a number or temporal string (date, time or date-time), the process returns False.
            - Temporal strings can not be compared based on their string representation due to the time zone /
            time-offset representations.

        Parameters
        ----------
        x : float or int or datetime.datetime
            First operand.
        y : float or int or datetime.datetime
            Second operand.

        Returns
        -------
        bool :
            Returns True if `x` is strictly lower than or equal to `y`, None if any operand is None, otherwise False.

        """
        if x is None or y is None:
            return None
        elif isinstance(x, str) and isinstance(y, str):
            return str2time(x) <= str2time(y)
        elif isinstance(x, (int, float, datetime.datetime)) and isinstance(y, (int, float, datetime.datetime)):
            return x <= y
        else:
            return False

    @staticmethod
    def exec_np(x, y, reduce=False):
        """
        Compares whether `x` is strictly lower than or equal to `y`.
        Remarks:
            - If any operand is None, the return value is None.
            - If any operand is not a number or temporal string (date, time or date-time), the process returns False.
            - Temporal strings can not be compared based on their string representation due to the time zone /
            time-offset representations.

        Parameters
        ----------
        x : np.ndarray
            First operand.
        y : np.ndarray
            Second operand.
        reduce : bool, optional
            If True, one value will be returned.
            If False, each value in `x` will be compared with the respective value in `y`. Defaults to False.

        Returns
        -------
        bool :
            Returns True if `x` is strictly lower than or equal to `y`, None if any operand is None, otherwise False.

        """
        if x is None or y is None:
            return None
        elif x.dtype.kind.lower() in ['f', 'i', 'm']:
            lte_ar = x <= y
            if reduce:
                return lte_ar.all()
            else:
                return lte_ar
        else:
            if reduce:
                return False
            else:
                return np.zeros(x.shape, dtype=np.bool)

    @staticmethod
    def exec_xar(x, y, reduce = False):
        """
        Compares whether `x` is strictly lower than or equal to `y`.
        Remarks:
            - If any operand is None, the return value is None.
            - If any operand is not a number or temporal string (date, time or date-time), the process returns False.
            - Temporal strings can not be compared based on their string representation due to the time zone /
            time-offset representations.

        Parameters
        ----------
        x : xr.DataArray, integer, float
            First operand.
        y : xr.DataArray, integer, float
            Second operand.
        reduce : bool, optional
            If True, one value will be returned.
            If False, each value in `x` will be compared with the respective value in `y`. Defaults to False.

        Returns
        -------
        bool :
            Returns True if `x` is strictly lower than or equal to `y`, None if any operand is None, otherwise False.

        """
        if x is None or y is None:
            return None
        elif type(x) in [int, float, xr.DataArray] and type(y) in [int, float, xr.DataArray]:
            lte_ar = x <= y
            lte_ar = keep_attrs(x, y, lte_ar)
            if reduce:
                return lte_ar.all()
            else:
                return lte_ar
        return False

    @staticmethod
    def exec_da():
        pass


########################################################################################################################
# Between Process
########################################################################################################################

@process
def between():
    """
    Returns class instance of `Between`.
    For more details, please have a look at the implementations inside `Between`.

    Returns
    -------
    Between :
        Class instance implementing all 'between' processes.

    """
    return Between()


class Between:
    """
    Class instance implementing all 'between' processes.

    """

    @staticmethod
    def exec_num(x, min, max, exclude_max=False):
        """
        By default this process checks whether `x` is greater than or equal to `min` and lower than or equal to `max`.
        All definitions from and_, gte and lte apply here as well. If `exclude_max` is set to True the upper bound is
        excluded so that the process checks whether `x` is greater than or equal to `min` and lower than `max`.
        Lower and upper bounds are not allowed to be swapped. `min` must be lower than or equal to `max` or otherwise
        the process always returns False.

        Parameters
        ----------
        x : float or int or datetime.datetime
            The value to check.
        min : float or int or datetime.datetime
            Lower boundary (inclusive) to check against.
        max : float or int or datetime.datetime
            Upper boundary (inclusive) to check against.
        exclude_max : bool, optional
            Exclude the upper boundary `max` if set to True. Defaults to False.

        Returns
        -------
        bool :
            True if `x` is between the specified bounds, otherwise False.

        """
        if x is None or min is None or max is None:
            return None

        if isinstance(x, str):
            x = str2time(x)

        if isinstance(min, str):
            min = str2time(min)

        if isinstance(max, str):
            max = str2time(max)

        if Lt().exec_num(max, min):
            return False

        if exclude_max:
            return Gte.exec_num(x, min) & Lt.exec_num(x, max)
        else:
            return Gte.exec_num(x, min) & Lte.exec_num(x, max)

    @staticmethod
    def exec_np(x, min, max, exclude_max=False, reduce=False):
        """
        By default this process checks whether `x` is greater than or equal to `min` and lower than or equal to `max`.
        All definitions from and_, gte and lte apply here as well. If `exclude_max` is set to True the upper bound is
        excluded so that the process checks whether `x` is greater than or equal to `min` and lower than `max`.
        Lower and upper bounds are not allowed to be swapped. `min` must be lower than or equal to `max` or otherwise
        the process always returns False.

        Parameters
        ----------
        x : np.ndarray
            The array values to check.
        min : float or int or datetime.datetime
            Lower boundary (inclusive) to check against.
        max : float or int or datetime.datetime
            Upper boundary (inclusive) to check against.
        exclude_max : bool, optional
            Exclude the upper boundary `max` if set to True. Defaults to False.
        reduce : bool, optional
            If True, one value will be returned.
            If False, each value in `x` evaluated and returned. Defaults to False.

        Returns
        -------
        bool or np.ndarray:
            True if `x` is between the specified bounds, otherwise False.

        """
        if x is None or min is None or max is None:
            return None

        min = np.array(min)  # cast to np.array because of datetime objects
        max = np.array(max)  # cast to np.array because of datetime objects

        if Lt().exec_num(max, min):
            return False

        if exclude_max:
            return Gte.exec_np(x, min, reduce=reduce) & Lt.exec_np(x, max, reduce=reduce)
        else:
            return Gte.exec_np(x, min, reduce=reduce) & Lte.exec_np(x, max, reduce=reduce)

    @staticmethod
    def exec_xar(x, min, max, exclude_max=False, reduce=False):
        """
        By default this process checks whether `x` is greater than or equal to `min` and lower than or equal to `max`.
        All definitions from and_, gte and lte apply here as well. If `exclude_max` is set to True the upper bound is
        excluded so that the process checks whether `x` is greater than or equal to `min` and lower than `max`.
        Lower and upper bounds are not allowed to be swapped. `min` must be lower than or equal to `max` or otherwise
        the process always returns False.

        Parameters
        ----------
        x : xr.DataArray
            The array values to check.
        min : float or int or datetime.datetime
            Lower boundary (inclusive) to check against.
        max : float or int or datetime.datetime
            Upper boundary (inclusive) to check against.
        exclude_max : bool, optional
            Exclude the upper boundary `max` if set to True. Defaults to False.
        reduce : bool, optional
            If True, one value will be returned.
            If False, each value in `x` evaluated and returned. Defaults to False.

        Returns
        -------
        bool or xr.DataArray:
            True if `x` is between the specified bounds, otherwise False.
        """
        if x is None or min is None or max is None:
            return None

        if Lt().exec_num(max, min):
            return False

        if exclude_max:
            bet = da.array.logical_and(Gte.exec_xar(x, min, reduce=reduce) , Lt.exec_xar(x, max, reduce=reduce))
        else:
            bet = da.array.logical_and(Gte.exec_xar(x, min, reduce=reduce) , Lte.exec_xar(x, max, reduce=reduce))
        if isinstance(x, xr.DataArray):
            bet.attrs = x.attrs
        return bet

    @staticmethod
    def exec_da():
        pass

