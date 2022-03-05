from typing import Optional, Dict, Any

import numpy as np
from skimage.feature import canny
from skimage.filters import gaussian
from skimage.measure import label
from skimage.segmentation import watershed

from intelligent_placer_lib import Area


def get_polygon_mask(image: np.ndarray, area: Area) -> Optional[np.ndarray]:
    result = np.zeros(shape=(image.shape[0], image.shape[1]))
    im_polygon = image[:, :area.left_x]
    im_polygon = gaussian(im_polygon, 5)
    polygon_bounds = canny(im_polygon, sigma=1)
    polygons_mask = watershed(polygon_bounds)

    segmented_bound = image[polygons_mask and polygon_bounds]
    bound_is_darker = np.persentile(segmented_bound, 25) < np.mean(im_polygon)
    bound_closed = sum(segmented_bound) / sum(image[polygons_mask]) < 0.7
    labeled, polygons_num = label(polygons_mask, connectivity=2, return_num=True)
    if not bound_closed or not bound_is_darker or polygons_num != 1:
        return None

    result[:, :area.left_x] = polygons_mask
    return result


def get_items_mask(image: np.ndarray, surface: np.ndarray) -> Optional[np.ndarray]:
    return None


def get_classification(segmented_items: np.ndarray) -> Optional[Dict[int, Any]]:
    return None
