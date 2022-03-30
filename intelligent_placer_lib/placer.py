from math import ceil
from typing import List, Tuple

import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.measure import regionprops


def estimate_possibility(figure_mask: np.ndarray, objects_masks: List[np.ndarray]) -> bool:
    if np.sum(figure_mask > 0) < sum([np.sum(object_mask > 0) for object_mask in objects_masks]):
        return False
    return True


def try_place(figure_mask: np.ndarray, object_mask: np.ndarray, start_x: int, start_y:int) -> Tuple[int, int]:
    x = start_x
    while x + object_mask.shape[1] < figure_mask.shape[1]:
        y = start_y
        while y + object_mask.shape[0] < figure_mask.shape[0]:
            cur_figure_area = figure_mask[y : y + object_mask.shape[0], x : x + object_mask.shape[1]]
            outliers = np.logical_and(np.logical_xor(cur_figure_area == 255, object_mask > 0), cur_figure_area != 255)
            if outliers.any():
                y += 1
            else:
                return x, y
        x += 1
    return -1, -1


def add_object(figure_mask: np.ndarray, object_mask: np.ndarray, start_x: int, start_y:int) -> np.ndarray:
    work_area = figure_mask[start_y: start_y + object_mask.shape[0], start_x: start_x + object_mask.shape[1]]
    for i in range(len(work_area)):
        for j in range(len(work_area[i])):
            if object_mask[i][j] > 0:
                work_area[i][j] = object_mask[i][j]
    return figure_mask


def rotate_mask(object_mask: np.ndarray, rotation: int):
    height, width = object_mask.shape
    y_cntr, x_cntr = height // 2, width // 2
    rotation_matrix = cv2.getRotationMatrix2D((y_cntr, x_cntr), -rotation, 1.0)
    cos_rotation, sin_rotation = np.abs(rotation_matrix[0][0]), np.abs(rotation_matrix[0][1])
    new_width = ceil((height * sin_rotation) + (width * cos_rotation))
    new_height = ceil((height * cos_rotation) + (width * sin_rotation))
    rotation_matrix[1][2] += (new_width / 2) - x_cntr
    rotation_matrix[0][2] += (new_height / 2) - y_cntr
    new_mask = cv2.warpAffine(object_mask, rotation_matrix, (new_width, new_height))
    prop = regionprops(new_mask)[0]
    return new_mask[prop.bbox[0]:prop.bbox[2], prop.bbox[1]: prop.bbox[3]]


def place_recursively(figure_mask: np.ndarray, objects_masks: List[np.ndarray], x_start, y_start,
                      rotations: bool) -> Tuple[bool, np.ndarray]:
    rotate = [0]
    if rotations:
        rotate = list(range(0, 360, 5))
    for degree in rotate:
        object_rotated = rotate_mask(objects_masks[0], degree)
        x = x_start
        cur_degree = True
        while x + object_rotated.shape[1] <= figure_mask.shape[1] and cur_degree:
            y = y_start
            while y + object_rotated.shape[0] <= figure_mask.shape[0] and cur_degree:
                placed_mask = figure_mask.copy()
                cur_x, cur_y = try_place(placed_mask, object_rotated, x, y)
                if cur_x == -1 or cur_y == -1:
                    cur_degree = False
                    break

                placed_mask = add_object(placed_mask, object_rotated, cur_x, cur_y)
                if len(objects_masks) == 1:
                    return True, placed_mask
                else:
                    result, placed_mask = place_recursively(placed_mask, objects_masks[1:], 0, 0, rotations)
                    if result:
                        return True, placed_mask
                y += 1
            x += 1
    return False, None


def place_objects(polygon_mask: np.ndarray, objects_masks: List[np.ndarray],
                  rotations: bool = False) -> Tuple[bool, np.ndarray]:
    if not estimate_possibility(polygon_mask, objects_masks):
        return False, None
    figure_mask = polygon_mask.copy()
    prop = regionprops(figure_mask)[0]
    figure_mask = figure_mask[prop.bbox[0]:prop.bbox[2], prop.bbox[1]: prop.bbox[3]]
    for object_mask in objects_masks:
        prop = regionprops(object_mask)[0]
        object_mask = object_mask[prop.bbox[0]:prop.bbox[2], prop.bbox[1]: prop.bbox[3]]
        if object_mask.shape[0] > figure_mask.shape[0] or object_mask.shape[0] > figure_mask.shape[0]:
            return False, None
    return place_recursively(figure_mask, objects_masks, 0, 0, rotations)
