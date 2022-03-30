import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import regionprops, label

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
    objects_masks = [segmented_items.copy()[prop.slice] * prop.image_filled
                     for prop in regionprops(label(segmented_items))]
    for i in range(min(TRY_PLACE_TIMES, len(objects_masks))):
        cur_sequence = np.random.permutation(objects_masks)
        result, placed = place_objects(polygon_mask, cur_sequence, rotations=True)
        if result:
            if verbose:
                plt.imshow(placed)
                plt.title('Placed objects.')
                plt.show()
            return result
    return False
