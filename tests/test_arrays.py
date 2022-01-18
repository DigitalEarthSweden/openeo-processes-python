"""
Most tests are in alignment with:
https://openeo.org/documentation/1.0/processes.html
"""

import numpy as np
import openeo_processes as oeop
import pytest
import xarray as xr
from unittest import TestCase
import math


@pytest.mark.usefixtures("test_data")
class ArrayTester(TestCase):
    """ Tests all array functions. """

    def test_array_create(self):
        """ Tests `array_create` function. """
        assert (oeop.array_create(2,3) == np.array([2,2,2])).all()
        assert (len(oeop.array_create([])) == 0)
        assert (oeop.array_create([1, 2, 3], repeat=2) == np.array([1,2,3,1,2,3])).all()
        assert (oeop.array_create(["A", "B", "C"]) == np.array(["A", "B", "C"])).all()
        assert len(oeop.array_create([np.nan, 1], 2) == 4)

    def test_array_modify(self):
        """ Tests `array_modify` function. """
        assert (oeop.array_modify(np.array([2,3]), np.array([4,7]), 1, 0) == np.array([2,4,7,3])).all()
        assert (oeop.array_modify(data = ["a","d","c"], values = ["b"], index = 1) == ['a','b','c']).all()
        assert (oeop.array_modify(data = ["a","b",4,5], values = [1,2,3], index = 0, length = 2) == [1,2,3,4,5]).all()
        assert (oeop.array_modify(data = ["a","c"], values = ["b"], index = 1, length = 0) == ["a","b","c"]).all()
        assert (oeop.array_modify(data = [np.nan,np.nan,"a","b","c"], values = [], index = 0, length = 2) == ["a","b","c"]).all()
        assert (oeop.array_modify(data=["a", "b", "c"], values=[], index=1, length=10) == ["a"])

    def test_array_concat(self):
        """ Tests `array_concat` function. """
        assert (oeop.array_concat(np.array([2, 3]), np.array([4, 7]))== np.array([2,3,4,7])).all()
        assert (oeop.array_concat(array1 = ["a","b"], array2 = [1,2]) == np.array(['a','b',1,2], dtype=object)).all()

    def test_array_contains(self):
        """ Tests `array_contains` function. """
        assert oeop.array_contains([1, 2, 3], value=2)
        assert not oeop.array_contains(["A", "B", "C"], value="b")
        assert not oeop.array_contains([1, 2, 3], value="2")
        assert oeop.array_contains([1, 2, 3], value=2)
        assert oeop.array_contains([1, 2, np.nan], value=np.nan)
        assert oeop.array_contains([[1, 2], [3, 4]], value=[1, 2])
        assert not oeop.array_contains([[1, 2], [3, 4]], value=2)
        assert oeop.array_contains([{"a": "b"}, {"c": "d"}], value={"a": "b"})
        assert oeop.array_contains(self.test_data.xr_data_factor(3, 5), value = 5)
        assert not oeop.array_contains(self.test_data.xr_data_factor(3, 5), value= 1)
        assert oeop.array_contains(self.test_data.xr_data_factor(np.nan, 5), value=np.NaN)
        assert oeop.array_contains(self.test_data.xr_data_factor(3, 5), value= [3, 5]).all()

    def test_array_element(self):
        """ Tests `array_element` function. """
        # numpy tests
        assert oeop.array_element([9, 8], label="B", labels=np.array(["A", "B"])) == 8
        assert oeop.array_element([9, 8, 7, 6, 5], index=2) == 7
        assert oeop.array_element(["A", "B", "C"], index=0) == "A"
        assert np.isnan(oeop.array_element([], index=0, return_nodata=True))

        # multi-dim
        test_array = np.empty((3, 2, 2))
        test_array[0, :, :] = np.array([[1, 2], [3, 4]])
        test_array[1, :, :] = np.array([[1, 2], [3, 4]]) * 20
        test_array[2, :, :] = np.array([[1, 2], [3, 4]]) * 500
        array_i2_d0 = np.array([
                                [500, 1000],
                                [1500, 2000]
                                ])
        array_i0_d1 = np.array([
                                [1, 2],
                                [20, 40],
                                [500, 1000]
                                ])
        assert np.isclose(oeop.array_element(test_array, index=2, dimension=0), array_i2_d0, equal_nan=True).all()
        assert np.isclose(oeop.array_element(test_array, index=0, dimension=1), array_i0_d1, equal_nan=True).all()

        # xarray tests
        xr.testing.assert_equal(
            oeop.array_element(self.test_data.xr_data_4d, dimension='bands', label="B08"),
            self.test_data.xr_data_3d)
        xr.testing.assert_equal(
            oeop.array_element(self.test_data.xr_data_4d, dimension='bands', index=0),
            self.test_data.xr_data_3d)
        # Assert raised errors?
        # ArrayElementNotAvailable
        # oeop.array_element(self.xr_data_4d, dimension='s', label="B09")
        # oeop.array_element(self.xr_data_4d, dimension='s', index=4)

    def test_count(self):
        """ Tests `count` function. """
        assert oeop.count([]) == 0
        assert oeop.count([], condition=True) == 0
        assert oeop.count([1, 0, 3, 2]) == 4
        assert oeop.count(["ABC", np.nan]) == 1
        assert oeop.count([False, np.nan], condition=True) == 2
        assert oeop.count([0, 1, 2, 3, 4, 5, np.nan], condition=oeop.gt, context={'y': 2}) == 3
        assert oeop.count([0, 1, 2, 3, 4, 5, np.nan], condition=oeop.lte, context={'y': 2}) == 3
        assert oeop.count(xr.DataArray(np.array([0, 1, 2, 3, 4, 5, np.nan])), condition=oeop.lte, context={'y': 2}) == 3
        assert oeop.count(xr.DataArray(np.array([0, 1, 2, 3, 4, 5, np.nan])), condition=oeop.gt, context={'y': 2}) == 3
        assert oeop.count(xr.DataArray(np.array([0, 1, 2, 3, 4, 5, np.nan])), condition=oeop.between, context={'min': 2, 'max': 4}) == 3
        assert oeop.count(self.test_data.xr_data_factor(3, 5), condition=oeop.gte, context={'y': 5}) == 15

    def test_array_apply(self):
        """ Tests `array_apply` function. """
        assert oeop.array_apply([1, 0, 3, 2], process=oeop.gt, context={'y':-1, 'reduce':True})
        assert oeop.array_apply(self.test_data.xr_data_factor(3, 5), process=oeop.gt, context={'y': 1, 'reduce': True})
        assert oeop.array_apply(self.test_data.xr_data_factor(3, 5), process=oeop.gt, context={'y': 1}).all()

    def test_array_filter(self):
        """ Tests `array_filter` function. """
        assert oeop.array_filter([1, 0, 3, 2], condition=oeop.lte, context={'y':0.5}) == 0
        assert (oeop.array_filter([1, 0, 3, 2, np.nan, 3], condition=oeop.gt, context={'y': 1}) == [3, 2, 3]).all()
        assert (oeop.array_filter([0, 1, 2, 3, 4, 5, np.nan], condition=oeop.gt, context={'y': 2}) == [3, 4, 5]).all()
        xr.testing.assert_equal(oeop.array_filter(self.test_data.xr_data_factor(3, 5), condition=oeop.lte, context={'y': 4}),
                                self.test_data.xr_data_factor(3, 5)[:1, :, :])

    def test_array_find(self):
        """ Tests `array_find` function. """
        assert oeop.array_find([1, 0, 3, 2], value= 3) == 2
        assert np.isnan(oeop.array_find([1, 0, 3, 2, np.nan, 3], value = np.nan))
        assert (oeop.array_find(self.test_data.xr_data_factor(3,5), value = 5) == 15)
        assert math.isnan(oeop.array_find(self.test_data.xr_data_factor(3,5), value = 4))
        assert (oeop.array_find(self.test_data.xr_data_factor(3,5), value = 5, dimension = 'time') == 1).all()
        assert (oeop.array_find(self.test_data.xr_data_factor(3, 5), value=3, dimension='time') == 0).all()
        assert (oeop.array_find(self.test_data.xr_data_factor(-3, -5), value=(-5), dimension='time') == 1).all()
        assert np.isnan(oeop.array_find(self.test_data.xr_data_factor(-3, -5), value=(0), dimension='time').values).all()

    def test_array_labels(self):
        """ Tests `array_labels` function. """
        assert (oeop.array_labels([1, 0, 3, 2]) == np.array([0, 1, 2, 3])).all()
        assert (oeop.array_labels([[1, 0, 3, 2]], dimension = 0) == np.array([0]))
        assert (oeop.array_labels(self.test_data.xr_data_factor(3,5), dimension = 0) == xr.DataArray(np.arange(2))).all()
        assert (oeop.array_labels(self.test_data.xr_data_factor(3,5), dimension = 'x') == xr.DataArray(np.arange(3))).all()
        assert (oeop.array_labels(self.test_data.xr_data_factor(3,5), dimension = 'y') == xr.DataArray(np.arange(5))).all()


    def test_first(self):
        """ Tests `first` function. """
        assert oeop.first([1, 0, 3, 2]) == 1
        assert oeop.first([np.nan, "A", "B"]) == "A"
        assert np.isnan(oeop.first([np.nan, 2, 3], ignore_nodata=False))
        assert np.isnan(oeop.first([]))

        # 2D test
        test_arr = np.array([[[np.nan, 2], [1, 2]], [[3, 2], [1, 2]], [[1, 2], [1, np.nan]]])
        first_elem_ref = np.array([[[3., 2.], [1., 2.]]])
        first_elem = oeop.first(test_arr)
        assert np.isclose(first_elem, first_elem_ref, equal_nan=True).all()
        test_arr = np.array([[[np.nan, 2], [1, 2]], [[3, 2], [1, 2]], [[1, 2], [1, np.nan]]])
        first_elem_ref = np.array([[[np.nan, 2.], [1., 2.]]])
        first_elem = oeop.first(test_arr, ignore_nodata=False)
        assert np.isclose(first_elem, first_elem_ref, equal_nan=True).all()
        xr.testing.assert_equal( oeop.first(self.test_data.xr_data_factor(3,5), dimension = 1, ignore_nodata = True) , self.test_data.xr_data_factor(3,5)[:,0,:])
        xr.testing.assert_equal( oeop.first(self.test_data.xr_data_factor(3,5), dimension = 'x', ignore_nodata = False) , self.test_data.xr_data_factor(3,5)[:,:,0])

    def test_last(self):
        """ Tests `last` function. """
        assert oeop.last([1, 0, 3, 2]) == 2
        assert oeop.last(["A", "B", np.nan]) == "B"
        assert np.isnan(oeop.last([0, 1, np.nan], ignore_nodata=False))
        assert np.isnan(oeop.last([]))

        # 2D test
        test_arr = np.array([[[np.nan, 2], [1, 2]], [[3, 2], [1, 3]], [[1, 2], [1, np.nan]]])
        last_elem_ref = np.array([[[1., 2.], [1., 3.]]])
        last_elem = oeop.last(test_arr)
        assert np.isclose(last_elem, last_elem_ref, equal_nan=True).all()
        test_arr = np.array([[[np.nan, 2], [1, 2]], [[3, 2], [1, 2]], [[1, 2], [1, np.nan]]])
        last_elem_ref = np.array([[[1., 2.], [1., np.nan]]])
        last_elem = oeop.last(test_arr, ignore_nodata=False)
        assert np.isclose(last_elem, last_elem_ref, equal_nan=True).all()
        xr.testing.assert_equal(oeop.last(self.test_data.xr_data_factor(3, 5), dimension='time', ignore_nodata=True),
                                self.test_data.xr_data_factor(3, 5)[-1, :, :])
        xr.testing.assert_equal(oeop.last(self.test_data.xr_data_factor(3, 5), dimension='x', ignore_nodata=False),
                                self.test_data.xr_data_factor(3, 5)[:, :, -1])
        assert (oeop.last(self.test_data.xr_data_factor(3, np.nan), dimension='time', ignore_nodata=True).values ==
                (self.test_data.xr_data_factor(3, 5)[0, :, :]).values).all()

    def test_order(self):
        """ Tests `order` function. """
        self.assertListEqual(oeop.order([6, -1, 2, np.nan, 7, 4, np.nan, 8, 3, 9, 9]).tolist(),
                             [1, 2, 8, 5, 0, 4, 7, 9, 10])
        self.assertListEqual(oeop.order([6, -1, 2, np.nan, 7, 4, np.nan, 8, 3, 9, 9], nodata=True).tolist(),
                             [1, 2, 8, 5, 0, 4, 7, 9, 10, 3, 6])
        self.assertListEqual(oeop.order([6, -1, 2, np.nan, 7, 4, np.nan, 8, 3, 9, 9], asc=False, nodata=True).tolist(),
                             [9, 10, 7, 4, 0, 5, 8, 2, 1, 3, 6])
        self.assertListEqual(oeop.order([6, -1, 2, np.nan, 7, 4, np.nan, 8, 3, 9, 9], asc=False, nodata=False).tolist(),
                             [6, 3, 9, 10, 7, 4, 0, 5, 8, 2, 1])
        assert (oeop.order(self.test_data.xr_data_factor(3, 5), dimension='time') == self.test_data.xr_data_factor(0, 1).values).all()
        assert (oeop.order(self.test_data.xr_data_factor(3, 5), dimension='time', asc=False) == self.test_data.xr_data_factor(1, 0).values).all()

    def test_rearrange(self):
        """ Tests `rearrange` function. """
        self.assertListEqual(oeop.rearrange([5, 4, 3], [2, 1, 0]).tolist(), [3, 4, 5])
        self.assertListEqual(oeop.rearrange([5, 4, 3, 2], [0, 2, 1, 3]).tolist(), [5, 3, 4, 2])
        self.assertListEqual(oeop.rearrange([5, 4, 3, 2], [1, 3]).tolist(), [4, 2])
        xr.testing.assert_equal(oeop.rearrange(self.test_data.xr_data_factor(3, 5), [1,0]),
                                xr.concat([self.test_data.xr_data_factor(3, 5)[1], self.test_data.xr_data_factor(3, 5)[0]], 'time'))

    def test_sort(self):
        """ Tests `sort` function. """
        self.assertListEqual(oeop.sort([6, -1, 2, np.nan, 7, 4, np.nan, 8, 3, 9, 9]).tolist(),
                             [-1, 2, 3, 4, 6, 7, 8, 9, 9])
        assert np.isclose(oeop.sort([6, -1, 2, np.nan, 7, 4, np.nan, 8, 3, 9, 9], asc=False, nodata=True),
                          [9, 9, 8, 7, 6, 4, 3, 2, -1, np.nan, np.nan], equal_nan=True).all()
        assert (oeop.sort(self.test_data.xr_data_factor(5, 3), dimension='time') == self.test_data.xr_data_factor(3, 5).values).all()
        assert (oeop.sort(self.test_data.xr_data_factor(3, 5), dimension='time', asc=False) == self.test_data.xr_data_factor(5, 3).values).all()

