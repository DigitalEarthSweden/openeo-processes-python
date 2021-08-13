from datetime import datetime

import numpy as np
import pytest
import xarray as xr
from affine import Affine
from datacube import Datacube
from datacube.model import Measurement
from datacube.utils.dates import mk_time_coord
from datacube.utils.geometry import GeoBox, CRS


# Taken from https://github.com/opendatacube/datacube-core/blob/develop/datacube/testutils/__init__.py#L429
# with minor adoptions
def mk_sample_xr_dataset(crs="EPSG:3578",
                         shape=(33, 74),
                         resolution=None,
                         xy=(0, 0),
                         time=None,
                         name='band',
                         dtype='int16',
                         nodata=np.nan,
                         units='1'):
    """ Note that resolution is in Y,X order to match that of GeoBox.

        shape (height, width)
        resolution (y: float, x: float) - in YX, to match GeoBox/shape notation

        xy (x: float, y: float) -- location of the top-left corner of the top-left pixel in CRS units
    """

    if isinstance(crs, str):
        crs = CRS(crs)

    if resolution is None:
        resolution = (-10, 10) if crs is None or crs.projected else (-0.01, 0.01)

    t_coords = {}
    if time is not None:
        t_coords['time'] = mk_time_coord(time)

    transform = Affine.translation(*xy)*Affine.scale(*resolution[::-1])
    h, w = shape
    geobox = GeoBox(w, h, transform, crs)

    storage = Datacube.create_storage(
        t_coords, geobox, [Measurement(name=name, dtype=dtype, nodata=nodata, units=units)])
    return storage.to_array(dim="bands")


@pytest.fixture(scope="class")
def test_data(request):
    class TestDataDriver:
        def __init__(self):
            self.steps = {'y': 5, 'x': 3}
            self.coords_4d = {
                'bands': ['B08', 'B04', 'B02'],
                'time': [datetime(2019, 12, 1), datetime(2019, 12, 5)],
                'y': np.arange(55.3, 55.3 + self.steps['y']),
                'x': np.arange(118.9, 118.9 + self.steps['x'])
            }
            self.coords_3d = {
                'time': [datetime(2019, 12, 1), datetime(2019, 12, 5)],
                'y': np.arange(55.3, 55.3 + self.steps['y']),
                'x': np.arange(118.9, 118.9 + self.steps['x'])
            }
            self._get_numpy()
            self._get_xarray()

        def _get_numpy(self):
            """
            Returns a fixed numpy array with 4 dimensions.
            """

            data = np.ones((3, 2, self.steps['y'], self.steps['x']))
            data[0, :] *= 8  # identify band 8 by its value
            data[1, :] *= 4  # identify band 4 by its value
            data[2, :] *= 2  # identify band 2 by its value

            data[:, 1, :] *= 10  # second t-step of each band multiplied by 10

            self.np_data_4d = data
            self.np_data_3d = data[0, :]

        def _get_xarray(self):
            """
            Returns a fixed xarray DataArray array with 3 labelled dimensions
            with coordinates.
            """

            self.xr_data_4d = xr.DataArray(data=self.np_data_4d,
                                           dims=self.coords_4d.keys(),
                                           coords=self.coords_4d)
            self.xr_data_4d.attrs['crs'] = 'EPSG:4326'
            self.xr_data_3d = xr.DataArray(data=self.np_data_3d,
                                           dims=self.coords_3d.keys(),
                                           coords=self.coords_3d)
            self.xr_data_3d.attrs['crs'] = 'EPSG:4326'
            self.xr_odc_data_3d = mk_sample_xr_dataset()
            self.xr_odc_data_4d = mk_sample_xr_dataset(
                time=['2020-02-13T11:12:13.1234567Z', '2020-02-14T11:12:13.1234567Z'])

        def xr_data_factor(self, factor_1=1.0, factor_2=1.0):
            data = np.ones((3, 2, self.steps['y'], self.steps['x']))
            data[0, 0] *= factor_1
            data[:, 1] *= factor_2
            xdata = xr.DataArray(data=data[0, :],
                                 dims=self.coords_3d.keys(),
                                 coords=self.coords_3d)
            xdata.attrs['crs'] = 'EPSG:4326'  # create a data array with variable values
            return xdata

    request.cls.test_data = TestDataDriver()
