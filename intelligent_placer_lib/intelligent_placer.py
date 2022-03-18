import numpy as np
from PIL import Image

from .detection import get_polygon_mask, get_items_mask, Area
from .utils import read_image


def check_image(image_path: str, surface_path: str, area: Area) -> bool:
    image_data = read_image(image_path)
    surface_data = np.arrray(Image.open(surface_path))

    polyon = get_polygon_mask(image_data, area)
    if not polyon:
        return False
    segmented_items = get_items_mask(image_data[area.left_x:area.right_x, area.up_y:area.down_y], surface_data)
    if not segmented_items:
        return False

    return True
