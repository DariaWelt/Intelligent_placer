from enum import Enum
from typing import List, Tuple

import cv2

from .detection import get_items_mask
from .utils import read_image


class PolygonMode(Enum):
    pixels = 'pixels'
    relative = 'relative'


def validate_input(im_size: Tuple[int, int], polygon: List[Tuple[float, float]], mode: str):
    if mode not in PolygonMode.__members__:
        raise ValueError('Unsupported polygon mode')
    if len(polygon) < 3:  # or is_one_line(polygon)
        raise ValueError('Polygon must have at least 3 angles')
    if min(polygon, key=lambda x: x[1])[1] < 0 or min(polygon, key=lambda x: x[0])[0] < 0:
        raise ValueError('Invalid polygon coordinates values: negative values are not supported')
    max_y, max_x = (1, 1) if mode == PolygonMode.relative.value else im_size
    if max(polygon, key=lambda x: x[1])[1] > max_y or max(polygon, key=lambda x: x[0])[0] > max_x:
        raise ValueError(f'Invalid polygon coordinates values: points are bigger than maximums: {(max_x, max_y)}.')


def check_image(image_path: str, polygon: List[Tuple[float, float]], mode: str = 'pixels') -> bool:
    image_data = read_image(image_path)
    validate_input((image_data.shape[0], image_data.shape[1]), polygon, mode)
    segmented_items, contours = get_items_mask(image_data)
    if not any(segmented_items > 0):
        return False
    if not sum([cv2.contourArea(cnt) for cnt in contours]) < cv2.contourArea(polygon):
        return False
    return True
