from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import xarray as xr

@dataclass
class Storage:
    crs: str
    resolution: Dict[str, float]


@dataclass
class MetadataProduct:
    name: str


@dataclass
class MetadataProperties:
    pass

@dataclass
class Metadata:
    product: MetadataProduct
    # properties: MetadataProperties


@dataclass
class Measurement:
    name: str
    units: str
    dtype: str
    nodata: Any
    aliases: List[str] = field(default_factory=list)
    extra_dim: str = None


@dataclass
class ExtraDimensions:
    name: str
    values: List[Union[int, float]]
    dtype: Literal['float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32',
                   'uint64', 'complex64', 'complex128']

@dataclass
class Product:
    name: str
    description: str
    storage: Storage
    metadata: Metadata
    measurements: List[Measurement]
    metadata_type: str = "eo3"
    extra_dimensions: Optional[List[ExtraDimensions]] = None


def create_product(data: xr.Dataset) -> Product:
    """Create a product definition form an xr.Dataset."""
    first_data_var = data.data_vars[list(data.data_vars.keys())[0]]

    # handle extra-dims
    extra_dims = list(set(first_data_var.dims).difference({'bands', 'y', 'x', 'time'}))
    if extra_dims:
        extra_dimensions = [
            ExtraDimensions(
                name=dim,
                values=getattr(first_data_var, dim).values.tolist(),
                dtype=str(getattr(first_data_var, dim).values.dtype),
            )
            for dim in extra_dims]

    measurements = [
        Measurement(
            name=name,
            dtype=str(msnt.dtype),
            nodata=-9999,  # no data value set to -9999 in save result
            units="",  # TODO not implemented - currently ignored!
            extra_dim=extra_dims[0] if extra_dims else None  # currently one a single extra dim is supported
        )
        for name, msnt in data.data_vars.items()
    ]
    # Use first data var to define storage - arbitray selection
    is_geographic = first_data_var.geobox.crs.geographic
    res = first_data_var.geobox.resolution
    if is_geographic:
        resolution = {"latitude": res[0], "longitude": res[1]}
    else:
        resolution = {"y": res[0], "x": res[1]}

    prod = Product(
        name="PLACEHOLDER_PRODCUT_NAME",
        description=f"Results of job PLACEHOLDER_JOB_ID.",
        storage=Storage(
            crs=first_data_var.geobox.crs.to_wkt(),
            resolution=resolution,
        ),
        metadata=Metadata(
            product=MetadataProduct(
                name="PLACEHOLDER_PRODCUT_NAME",
            ),
        ),
        measurements=measurements,
        extra_dimensions=None
        #extra_dimensions if extra_dims else None,
    )
    return prod


def get_prod_dict(data: xr.Dataset) -> dict:
    """Create a product definition from an xr.Dataset and return it as dict."""
    product = create_product(data)
    prod_dict = asdict(product)
    if not prod_dict["extra_dimensions"]:
        prod_dict.pop("extra_dimensions")
        for idx in range(len(prod_dict["measurements"])):
            prod_dict["measurements"][idx].pop("extra_dim")
    return prod_dict
