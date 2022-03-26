from typing import List, Tuple

import numpy as np
from skimage.measure import regionprops


def try_place(figure_mask: np.ndarray, object_mask: np.ndarray, start_x: int, start_y:int) -> Tuple[int, int]:
    x = start_x
    while x + object_mask.shape[1] < figure_mask.shape[1]:
        y = start_y
        while y + object_mask.shape[0] < figure_mask.shape[0]:
            cur_figure_area = figure_mask[y : y + object_mask.shape[0], x : x + object_mask.shape[1]]
            outliers = np.logical_and(np.logical_xor(cur_figure_area > 0, object_mask == 0), cur_figure_area == 0)
            if outliers.any():
                y += 1
            else:
                return x, y
        x += 1
    return -1, -1


def add_object(figure_mask: np.ndarray, object_mask: np.ndarray, start_x: int, start_y:int) -> np.ndarray:
    work_area = figure_mask[start_y: start_y + object_mask.shape[0], start_x: start_x + object_mask.shape[1]]
    max_label = figure_mask.max()
    for i in range(len(work_area)):
        for j in range(len(work_area[i])):
            if object_mask[i][j] > 0:
                work_area[i][j] = max_label + 1
    return figure_mask


def place_recursively(figure_mask: np.ndarray, objects_masks: List[np.ndarray], x_start, y_start) -> Tuple[bool, np.ndarray]:
    x = x_start
    placed_mask = figure_mask.copy()
    while x + objects_masks[0].shape[1] <= figure_mask.shape[1]:
        y = y_start
        while y + objects_masks[0].shape[0] <= figure_mask.shape[0]:
            cur_x, cur_y = try_place(figure_mask, objects_masks[0], x, y)
            if cur_x == -1 or cur_y == -1:
                return False, None

            placed_mask = add_object(placed_mask, objects_masks[0], cur_x, cur_y)
            if len(objects_masks) == 1:
                return True, placed_mask
            else:
                result, mask = place_recursively(figure_mask, objects_masks[1:], 0, 0)
                if result:
                    return True, mask
            y += 1
        x += 1
    return False, None


def place_objects(polygon_mask: np.ndarray, objects_masks: List[np.ndarray]) -> Tuple[bool, np.ndarray]:
    figure_mask = polygon_mask.copy()
    prop = regionprops(figure_mask)[0]
    figure_mask = figure_mask[prop.bbox[0]:prop.bbox[2], prop.bbox[1]: prop.bbox[3]]
    for object_mask in objects_masks:
        prop = regionprops(object_mask)[0]
        object_mask = object_mask[prop.bbox[0]:prop.bbox[2], prop.bbox[1]: prop.bbox[3]]
        if object_mask.shape[0] > figure_mask.shape[0] or object_mask.shape[0] > figure_mask.shape[0]:
            return False, None
    return place_recursively(figure_mask, objects_masks, 0, 0)
