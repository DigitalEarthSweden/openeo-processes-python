from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

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


@dataclass
class Product:
    name: str
    description: str
    storage: Storage
    metadata: Metadata
    measurements: List[Measurement]
    metadata_type: str = "eo3"


def create_product(data: xr.Dataset) -> Product:
    """Create a product definition form an xr.Dataset."""
    measurements = [
        Measurement(
            name=name,
            dtype=str(msnt.dtype),
            nodata=-9999,  # no data value set to -9999 in save result
            units="",  # TODO not implemented - currently ignored!
        )
        for name, msnt in data.data_vars.items()
    ]
    # Use first data var to define storage - arbitray selection
    first_data_var = data.data_vars[measurements[0].name]
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
    )
    return prod


def get_prod_dict(data: xr.Dataset) -> dict:
    """Create a product definition from an xr.Dataset and return it as dict."""
    product = create_product(data)
    return asdict(product)
