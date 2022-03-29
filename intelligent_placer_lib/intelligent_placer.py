from enum import Enum
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import regionprops

from intelligent_placer_lib.detection import get_items_mask
from intelligent_placer_lib.placer import place_objects
from intelligent_placer_lib.utils import read_image


TRY_PLACE_TIMES = 1


def check_image(image_path: str, verbose: bool = False) -> bool:
    image_data = read_image(image_path)
    polygon_contour, segmented_items = get_items_mask(image_data, verbose)
    if not (segmented_items > 0).any():
        return False
    polygon_mask = np.zeros((image_data.shape[0], image_data.shape[1]))
    cv2.drawContours(polygon_mask, [polygon_contour], -1, 255, cv2.FILLED)
    polygon_mask = polygon_mask.astype('uint8')
    segmented_items = segmented_items.astype('uint8')
    for i in range(TRY_PLACE_TIMES):
        objects_masks = [segmented_items.copy()[prop.bbox[0]:prop.bbox[2], prop.bbox[1]: prop.bbox[3]]
                         for prop in regionprops(segmented_items)]
        result, placed = place_objects(polygon_mask, objects_masks, rotations=True)
        if result:
            if verbose:
                plt.imshow(placed)
                plt.title('Placed objects.')
                plt.show()
            return result
    return False
