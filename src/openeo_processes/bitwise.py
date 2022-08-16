import numpy as np
import xarray as xr
from pandas import array

from openeo_processes.utils import process

########################################################################################################################
# Bitwise And Process
########################################################################################################################

# ---------------------------------------------------------------------------------
#                                   bitwise_operators
# ---------------------------------------------------------------------------------
def bitwise_operators(x, y, op):
    """
    x = None | int | xr.DataArray(int|float|object) | nd.ndarray(int|float|object) | daskarrat(int|float|object)
    y = None | int | xr.DataArray(int|float|object) | nd.ndarray(int|float|object) | daskarray(int|float|object)
    if x is array. y shall have the same shape.
    RETURNS
        None | array(int) |array(float)
        If x or y is float
        x operator y for all elements ex,ey in x,y.

    """
    # ---------------------------------------------------
    # INTERNAL                  mask_output
    # ---------------------------------------------------
    def mask_output(mask, data, no_data):
        if isinstance(data, xr.core.dataarray.DataArray):
            mask = mask == False  # invert the mask
            data = data.where(mask, no_data)
        else:
            # np and dask supports faster way
            data[mask] = no_data
        return data

    # ---------------------------------------------------
    # INTERNAL                 to_int_array
    # ---------------------------------------------------
    def to_int_array(a, a_type):
        """
        float[] may contain nans,
        and need to be converted to int to be
        used with the bitwise operators.
        object[] may contain None:s. They need to be
        converted to float and THEN to int.
        """
        if a_type == int:
            return a
        elif a_type == float:
            return a.astype(int)
        elif a_type == object:
            return a.astype(float).astype(int)
        raise Exception(
            f"Wrong datatype ({a_type}) "
            " provided to bitwise operator. Only "
            "int,float and object accepted as array arguments."
        )

    # ---------------- FUNCTION START (bitwise_operators) -----------------
    # Determine input types once for all (dtype,shape is used for avoiding xarray and daskarray imports)
    x_is_array = hasattr(x, "dtype") and x.shape != ()
    x_type = x.dtype if hasattr(x, "dtype") else type(x)

    y_is_array = hasattr(y, "dtype") and y.shape != ()
    y_type = y.dtype if hasattr(y, "dtype") else type(y)

    # Fasttrack for int input, both array and singles.
    # We don not need to mask anything
    # Broadcasting also works out of the box
    if x_type == int and y_type == int:
        if op == "&":
            res = x & y
        elif op == "|":
            res = x | y
        elif op == "^":
            res = x ^ y
        else:
            raise Exception(f"Unknown operator {op}.")
        return res
    # The rest handles the cases where we have gaps in the data
    no_data = (
        {"value": None, "type": object}
        if x_type == object or x is None or y_type == object or y is None
        else {"value": np.nan, "type": float}
    )

    # Special case, if any of x or y is single none or nan
    if x_is_array == False and (x is None or x != x):
        return (
            np.full_like(y, no_data["value"], dtype=no_data["type"])
            if y_is_array
            else no_data["value"]
        )
    elif y_is_array == False and (y is None or y != y):
        return (
            np.full_like(x, no_data["value"], dtype=no_data["type"])
            if x_is_array
            else no_data["value"]
        )

    if x_type not in [int, float, object] or y_type not in [int, float, object]:
        raise Exception(
            f"Wrong datatype ({str(x_type)},{str(y_type)})"
            " provided to bitwise operator. Only "
            "int,float and object accepted."
        )

    # Bitwise operator only works on int (float and object only used to signal no data)
    x_int = to_int_array(x, x_type) if x_is_array else int(x)
    y_int = to_int_array(y, y_type) if y_is_array else int(y)

    if op == "&":
        res = x_int & y_int
    elif op == "|":
        res = x_int | y_int
    elif op == "^":
        res = x_int ^ y_int
    else:
        raise Exception(f"Unknown operator {op}.")

    # Special case of valid float,single value -> float
    if not hasattr(res, "astype"):
        return float(res)

    # Adjust return type to accomodate the no_data propagation
    if x_type == object or y_type == object:
        res = res.astype(object)
    else:
        res = res.astype(float)
    # Mask out any np.nans
    if x_is_array:
        x_mask = (x == None) | (x != x)

        res = mask_output(mask=x_mask, data=res, no_data=no_data["value"])
    if y_is_array:
        y_mask = (y == None) | (y != y)
        res = mask_output(mask=y_mask, data=res, no_data=no_data["value"])

    return res


@process
def bitwise_and():
    """
    Returns class instance of `Bitwise_And`.
    For more details, please have a look at the implementations inside `Bitwise_And`.

    Returns
    -------
    And :
        Class instance implementing all 'and' processes.

    """
    return Bitwise_And()


class Bitwise_And:
    """
    Class implementing all 'bitwise_and' processes.

    """

    @staticmethod
    def exec_num(x, y):
        """
        Returns bitwise and between two integers
        Parameters
        ----------
        x : int
        y : int

        Returns
        -------
        int: x & y
        """
        if x is None or np.nan(x) or y is None or np.isnan(y):
            return None
        return x & y

    @staticmethod
    def exec_np(x, y):
        """
        Returns bitwise-and between corresponding elements of x and y. x and y are equally shaped arrays or a single value that is broadcast over an array.
        Parameters
        ----------
        x : int or np.ndarray(int)
        y : int or np.ndarray(int)


        Returns
        -------
        np.array(int)

        """
        return bitwise_operators(x, y, "&")

    @staticmethod
    def exec_xar(x, y):
        """
        Returns bitwise-and between corresponding elements of x and y. x and y are equally shaped arrays or a single value that is broadcast over an array.

        Parameters
        ----------
        x : int | xr.DataArray(int)
        y : int | xr.DataArray(int)


        Returns
        -------
        xr.DataArray(int) :
            [ ex & ey if not isnan(ex) and not isnan(ey) else NaN for (ex,ey) in zip(x,y) ]
        """
        return bitwise_operators(x, y, "&")

    @staticmethod
    def exec_da(x, y):
        return bitwise_operators(x, y, "&")


"""
# DEBUG
ctor = np.array
size = 10

a = np.array(np.random.random(size))
b = np.array(np.random.random(size))

a_int = a.astype(int)
b_int = b.astype(int)
a_obj = a.astype(object)
b_obj = b.astype(object)

nan_i_a = np.random.randint(1, len(a), size=100)
a[nan_i_a] = None  # Becomes nan automatically
a_obj[nan_i_a] = None

nan_i_b = np.random.randint(1, len(b), size=100)
b[nan_i_b] = np.nan
b_obj[nan_i_b] = None

# Instead of creating several functions
testdata_x = {
    "None": None,
    "int": a_int[0],
    "int[]": a_int,
    "float[]": a,
    "float64": a[0],
    "float64-nan": a[nan_i_a][0],
    "float": 1.0,
    "float-nan": np.nan,
    "object[]": a_obj,
    "nan_idx": nan_i_a,
}
testdata_y = {
    "None": None,
    "int": b_int[0],
    "int[]": b_int,
    "float[]": b,
    "float64": b[0],
    "float64-nan": b[nan_i_a][0],
    "float": 1.0,
    "float-nan": np.nan,
    "object[]": b_obj,
    "nan_idx": nan_i_b,
}
bitwise_operators(testdata_x["float64-nan"], testdata_y["float[]"], op="&")
"""
