import os
import yaml
import xarray as xr

from openeo_processes.extension.product_model import get_prod_dict


def write_odc_product(dataset: xr.Dataset, output_filepath_data: str):
    """Create ODC product definition.

    Uses properties of xr.Dataset to define product.
    """
    folder_path = os.path.dirname(output_filepath_data)
    product_filepath = os.path.join(folder_path, "product.yml")
    product = get_prod_dict(dataset)
    with open(product_filepath, "w") as product_file:
        yaml.dump(product, product_file)
