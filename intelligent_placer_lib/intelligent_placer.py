from enum import Enum
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import regionprops

from .detection import get_items_mask
from .placer import place_objects
from .utils import read_image


TRY_PLACE_TIMES = 1


def check_image(image_path: str, verbose: bool = False) -> bool:
    image_data = read_image(image_path)
    polygon_contour, segmented_items = get_items_mask(image_data, verbose)
    if not (segmented_items > 0).any():
        return False
    polygon_mask = np.zeros((image_data.shape[0], image_data.shape[1]))
    cv2.drawContours(polygon_mask, [polygon_contour], -1, 255, cv2.FILLED)
    for i in range(TRY_PLACE_TIMES):
        objects_masks = [segmented_items.copy()[prop.bbox[0]:prop.bbox[2], prop.bbox[1]: prop.bbox[3]] for prop in regionprops(segmented_items)]
        result, placed = place_objects(polygon_mask, objects_masks)
        if result:
            if verbose:
                plt.imshow(placed)
                plt.show()
                plt.title('Placed objects')
            return result
    return False
