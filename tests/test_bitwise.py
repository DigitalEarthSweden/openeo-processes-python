import pytest
import numpy as np
import xarray as xr
import dask.array as da
import pandas as pd
import openeo_processes as oeop

# ---------------------------------------------------------------------------------
#                                   get_type_expectation
# ---------------------------------------------------------------------------------
def get_type_expectation(x_type, y_type):

    type_expextations = [
        {"x": "None", "y": "None", "exp": "None"},
        {"x": "None", "y": "float", "exp": "None"},
        {"x": "None", "y": "float-nan", "exp": "None"},
        {"x": "None", "y": "float64", "exp": "None"},
        {"x": "None", "y": "float64-nan", "exp": "None"},
        {"x": "None", "y": "int", "exp": "None"},
        {"x": "None", "y": "int[]", "exp": "object[]"},
        {"x": "None", "y": "float[]", "exp": "object[]"},
        {"x": "None", "y": "object[]", "exp": "object[]"},
        {"x": "float", "y": "None", "exp": "None"},
        {"x": "float", "y": "float", "exp": "float"},
        {"x": "float", "y": "float-nan", "exp": "float"},
        {"x": "float", "y": "float64", "exp": "float"},
        {"x": "float", "y": "float64-nan", "exp": "float"},
        {"x": "float", "y": "int", "exp": "float"},
        {"x": "float", "y": "int[]", "exp": "float[]"},
        {"x": "float", "y": "float[]", "exp": "float[]"},
        {"x": "float", "y": "object[]", "exp": "object[]"},
        {"x": "float-nan", "y": "None", "exp": "None"},
        {"x": "float-nan", "y": "float", "exp": "float"},
        {"x": "float-nan", "y": "float-nan", "exp": "float"},
        {"x": "float-nan", "y": "float64", "exp": "float"},
        {"x": "float-nan", "y": "float64-nan", "exp": "float"},
        {"x": "float-nan", "y": "int", "exp": "float"},
        {"x": "float-nan", "y": "int[]", "exp": "float[]"},
        {"x": "float-nan", "y": "float[]", "exp": "float[]"},
        {"x": "float-nan", "y": "object[]", "exp": "object[]"},
        {"x": "float64", "y": "None", "exp": "None"},
        {"x": "float64", "y": "float", "exp": "float"},
        {"x": "float64", "y": "float-nan", "exp": "float"},
        {"x": "float64", "y": "float64", "exp": "float"},
        {"x": "float64", "y": "float64-nan", "exp": "float"},
        {"x": "float64", "y": "int", "exp": "float"},
        {"x": "float64", "y": "int[]", "exp": "float[]"},
        {"x": "float64", "y": "float[]", "exp": "float[]"},
        {"x": "float64", "y": "object[]", "exp": "object[]"},
        {"x": "float64-nan", "y": "None", "exp": "None"},
        {"x": "float64-nan", "y": "float", "exp": "float"},
        {"x": "float64-nan", "y": "float-nan", "exp": "float"},
        {"x": "float64-nan", "y": "float64", "exp": "float"},
        {"x": "float64-nan", "y": "float64-nan", "exp": "float"},
        {"x": "float64-nan", "y": "int", "exp": "float"},
        {"x": "float64-nan", "y": "int[]", "exp": "float[]"},
        {"x": "float64-nan", "y": "float[]", "exp": "float[]"},
        {"x": "float64-nan", "y": "object[]", "exp": "object[]"},
        {"x": "int", "y": "None", "exp": "None"},
        {"x": "int", "y": "float", "exp": "float"},
        {"x": "int", "y": "float-nan", "exp": "float"},
        {"x": "int", "y": "float64", "exp": "float"},
        {"x": "int", "y": "float64-nan", "exp": "float"},
        {"x": "int", "y": "int", "exp": "int"},
        {"x": "int", "y": "int[]", "exp": "int[]"},
        {"x": "int", "y": "float[]", "exp": "float[]"},
        {"x": "int", "y": "object[]", "exp": "object[]"},
        {"x": "int[]", "y": "None", "exp": "object[]"},
        {"x": "int[]", "y": "float", "exp": "float[]"},
        {"x": "int[]", "y": "float-nan", "exp": "float[]"},
        {"x": "int[]", "y": "float64", "exp": "float[]"},
        {"x": "int[]", "y": "float64-nan", "exp": "float[]"},
        {"x": "int[]", "y": "int", "exp": "int[]"},
        {"x": "int[]", "y": "int[]", "exp": "int[]"},
        {"x": "int[]", "y": "float[]", "exp": "float[]"},
        {"x": "int[]", "y": "object[]", "exp": "object[]"},
        {"x": "float[]", "y": "None", "exp": "object[]"},
        {"x": "float[]", "y": "float", "exp": "float[]"},
        {"x": "float[]", "y": "float-nan", "exp": "float[]"},
        {"x": "float[]", "y": "float64", "exp": "float[]"},
        {"x": "float[]", "y": "float64-nan", "exp": "float[]"},
        {"x": "float[]", "y": "int", "exp": "float[]"},
        {"x": "float[]", "y": "int[]", "exp": "float[]"},
        {"x": "float[]", "y": "float[]", "exp": "float[]"},
        {"x": "float[]", "y": "object[]", "exp": "object[]"},
        {"x": "object[]", "y": "None", "exp": "object[]"},
        {"x": "object[]", "y": "float", "exp": "object[]"},
        {"x": "object[]", "y": "float-nan", "exp": "object[]"},
        {"x": "object[]", "y": "float64", "exp": "object[]"},
        {"x": "object[]", "y": "float64-nan", "exp": "object[]"},
        {"x": "object[]", "y": "int", "exp": "object[]"},
        {"x": "object[]", "y": "int[]", "exp": "object[]"},
        {"x": "object[]", "y": "float[]", "exp": "object[]"},
        {"x": "object[]", "y": "object[]", "exp": "object[]"},
    ]

    for test in type_expextations:
        if test["x"] == x_type and test["y"] == y_type:
            return test["exp"]
    raise Exception("Unknown in data type combination", x_type, y_type)


# ---------------------------------------------------------------------------------
#                              operators_reference_impl
# ---------------------------------------------------------------------------------


def operators_ref_impl(x, y, operator):
    x_is_array = hasattr(x, "dtype") and x.shape != ()
    x_type = x.dtype if hasattr(x, "dtype") else type(x)

    y_is_array = hasattr(y, "dtype") and y.shape != ()
    y_type = y.dtype if hasattr(y, "dtype") else type(y)

    if None in [x, y]:
        return None
    if np.nan in [x, y]:
        return np.nan

    op = lambda x, y: 123456
    if operator == "&":
        op = lambda x, y: x & y
    elif operator == "|":
        op = lambda x, y: x | y
    elif operator == "^":
        op = lambda x, y: x ^ y
    else:
        assert operator == "something wierd"

    if not x_is_array and not y_is_array:
        return op(x, y)

    if x_type == int and y_type == int:

        if not x_is_array and y_is_array:
            return op(x, y)

    X = x
    Y = y
    if not x_is_array:
        X = np.full_like(Y, X, dtype=object)
    elif not y_is_array:
        Y = np.full_like(X, Y, dtype=object)
    X_int = X.astype(float).astype(int)
    Y_int = Y.astype(float).astype(int)

    res = [
        op(X_int[i], Y_int[i]) if not pd.isna(X[i]) and not pd.isna(Y[i]) else None
        for i, _ in enumerate(X_int)
    ]

    return res


# ---------------------------------------------------------------------------------
#                                   check_result
# ---------------------------------------------------------------------------------
def check_result(x_type, y_type, test_data_x, test_data_y, operator, res):
    def is_array(a):
        return hasattr(a, "dtype") and a.shape != ()

    def get_type(a):
        return a.dtype if hasattr(a, "dtype") else type(a)

    def check_nans(dtype, idxs, res):
        if dtype == "object[]" or dtype == "float[]":
            for i in idxs:
                assert pd.isna(res[i])

    def check_array_result(X, Y, operator):
        # At least one of X,Y is an array
        # Make both arrays
        if not is_array(X):
            X = np.full_like(Y, X, dtype=object)
        if not is_array(Y):
            Y = np.full_like(X, Y, dtype=object)
        X_int = X.astype(float).astype(int)
        Y_int = Y.astype(float).astype(int)
        op = lambda x, y: 123456
        if operator == "&":
            op = lambda x, y: x & y
        elif operator == "|":
            op = lambda x, y: x | y
        elif operator == "^":
            op = lambda x, y: x ^ y
        else:
            assert operator == "something wierd"
        expected_result = [
            op(X_int[i], Y_int[i]) if not pd.isna(X[i]) and not pd.isna(Y[i]) else None
            for i, _ in enumerate(X_int)
        ]
        for r, e in zip(res, expected_result):
            if pd.isna(e):
                assert pd.isna(r)
            else:
                assert r == e

    out_type = get_type_expectation(x_type, y_type)

    if out_type == "None":
        assert res == None
    elif out_type == "object[]":
        assert is_array(res)
        assert res.dtype == object
        check_nans(x_type, test_data_x["nan_idx"], res)
        check_nans(y_type, test_data_y["nan_idx"], res)
        if x_type == "None":
            assert all(res == None)
        else:
            X = test_data_x[x_type]
            Y = test_data_y[y_type]
            check_array_result(X, Y, operator)
        if y_type == "None":
            assert all(res == None)

    elif out_type == "int":
        assert get_type(res) == int
    elif out_type == "float":
        get_type(res) == float
    elif out_type == "int[]":
        assert is_array(res)
        assert get_type(res) == int
        # No nans for int values.
    elif out_type == "float[]":
        assert is_array(res)
        assert get_type(res) == float
        check_nans(x_type, test_data_x["nan_idx"], res)
        check_nans(y_type, test_data_y["nan_idx"], res)
        if "nan" in x_type:
            assert all(pd.isna(res))
        if "nan" in y_type:
            assert all(pd.isna(res))
        X = test_data_x[x_type]
        Y = test_data_y[y_type]
        check_array_result(X, Y, operator)

    else:
        raise Exception(
            "Unknown out data type to expect", out_type
        )  # This should not happpen, remove it when finished.


# ---------------------------------------------------------------------------
#                                generate_data
# ---------------------------------------------------------------------------
def generate_data_3d(ctor):
    size = 10
    if ctor != da.from_array:
        a = ctor(np.random.random(size))
        b = ctor(np.random.random(size))
    else:
        # Dask arrays cannot be manipulated
        # So we need to take a detour here:
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
    if ctor == da.from_array:
        a = da.from_array(a, chunks=(1000))
        b = da.from_array(b, chunks=(1000))
        a_int = da.from_array(a_int, chunks=(1000))
        b_int = da.from_array(b_int, chunks=(1000))
        a_obj = da.from_array(a_obj, chunks=(1000))
        b_obj = da.from_array(b_obj, chunks=(1000))
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
    return testdata_x, testdata_y


def generate_data(ctor):
    size = 10
    if ctor != da.from_array:
        a = ctor(np.random.random(size))
        b = ctor(np.random.random(size))
    else:
        # Dask arrays cannot be manipulated
        # So we need to take a detour here:
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
    if ctor == da.from_array:
        a = da.from_array(a, chunks=(1000))
        b = da.from_array(b, chunks=(1000))
        a_int = da.from_array(a_int, chunks=(1000))
        b_int = da.from_array(b_int, chunks=(1000))
        a_obj = da.from_array(a_obj, chunks=(1000))
        b_obj = da.from_array(b_obj, chunks=(1000))
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
    return testdata_x, testdata_y


# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ctorname,ctor",
    [("numpy", np.array), ("xarray", xr.DataArray), ("daskarray", da.from_array)],
)

# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x_type",
    [
        ("None"),
        ("int"),
        ("int[]"),
        ("float"),
        ("float-nan"),
        ("float64"),
        ("float64-nan"),
        ("float[]"),
        ("object[]"),
    ],
)

# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "y_type",
    [
        ("None"),
        ("int"),
        ("int[]"),
        ("float"),
        ("float-nan"),
        ("float64"),
        ("float64-nan"),
        ("float[]"),
        ("object[]"),
    ],
)
@pytest.mark.parametrize(
    "operator",
    [
        ("&"),
        ("|"),
        ("^"),
    ],
)
# ---------------------------------------------------------------------------


def test_bitwise(ctor, ctorname, x_type, y_type, operator):
    test_data_x, test_data_y = generate_data(ctor)
    if ctorname == "daskarray" and "object" in x_type or "object" in y_type:
        # Daskarray does not support object arrays anyway
        return

    res = oeop.bitwise_operators(test_data_x[x_type], test_data_y[y_type], op=operator)
    # Check that the holes are there:
    check_result(x_type, y_type, test_data_x, test_data_y, operator, res)


def test_bitwise_3d_xarray():
    t1 = [[0b00010, 0b00101], [0b00101, 0b00010]]
    t2 = [
        [
            0b00101,
            np.nan,
        ],
        [
            0b00010,
            0b00101,
        ],
    ]

    # XARRAY
    x = xr.DataArray([t1, t2])
    # Correct:
    res = 2 & x.astype(float).astype(int)
    res = res.astype(float)
    res = res.where(x == x, 123)

    eo_res = oeop.bitwise_operators(x=2, y=x, op="&")
    eo_res = eo_res.where(eo_res == eo_res, 123)
    assert (res == eo_res).all()


def test_bitwise_3d_numpy_array():
    t1 = [[0b00010, 0b00101], [0b00101, 0b00010]]
    t2 = [
        [
            0b00101,
            np.nan,
        ],
        [
            0b00010,
            0b00101,
        ],
    ]

    x = np.array([t1, t2])
    # Correct:
    res = 2 & x.astype(float).astype(int)
    res = res.astype(float)
    res[x != x] = 123

    eo_res = oeop.bitwise_operators(x=2, y=x, op="&")
    eo_res[eo_res != eo_res] = 123
    assert (res == eo_res).all()
