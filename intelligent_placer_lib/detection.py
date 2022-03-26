from typing import Optional, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import gaussian, sobel

from .descriptor import get_descriptor, Point, match_descriptors
from .utils import to_grayscale, to_uint8_image

MIN_OBJECT_WIDTH: int = 50
MIN_OBJECT_AREA: int = MIN_OBJECT_WIDTH * MIN_OBJECT_WIDTH


def _get_objects_contours(image: np.ndarray, verbose: bool = False) -> List:
    source = to_grayscale(gaussian(image, int(MIN_OBJECT_WIDTH / 10)))
    bounds = sobel(source)
    if verbose:
        plt.imshow(bounds)
        plt.title('image bound')
        plt.show()
    _, bounds = cv2.threshold(bounds, np.percentile(bounds, 95), bounds.max(), 0)
    contours, hierarchy = cv2.findContours(to_uint8_image(bounds), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    im = np.zeros((bounds.shape[0], bounds.shape[1], 3))
    cv2.drawContours(im, contours, -1, (0, 255, 0))
    if verbose:
        plt.imshow(im)
        plt.title('image contours')
        plt.show()
    return list(filter(lambda contour: cv2.contourArea(contour) >= MIN_OBJECT_AREA, contours))


def _get_transformed_mask(segmented: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray, source_mask: np.ndarray,
                          object_class: int) -> Optional[np.ndarray]:
    src_key_pts = np.float32([src_pts]).reshape(-1, 1, 2)
    dst_key_pts = np.float32(dst_pts).reshape(-1, 1, 2)

    matrix, mask = cv2.findHomography(src_key_pts, dst_key_pts, cv2.RANSAC, 5.0)
    if matrix is None:
        return None

    p = np.where(source_mask >= 200)
    mask_points = [[point[1], point[0]] for point in zip(*p)]
    new_mask_points = cv2.perspectiveTransform(np.float32(mask_points).reshape(-1, 1, 2), matrix)[:, 0, :]
    for point in new_mask_points:
        if segmented.shape[0] <= int(point[1]) or int(point[1]) < 0 \
                or segmented.shape[1] <= int(point[0]) or int(point[0]) < 0:
            return None
        if segmented[int(point[1])][int(point[0])] != 0 and segmented[int(point[1])][int(point[0])] != object_class:
            return None
        segmented[int(point[1])][int(point[0])] = object_class
    plt.imshow(segmented + 230)
    plt.show()
    print(new_mask_points)
    return segmented


def _classify_objects(contours, image: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, List]:
    from .utils import items_info

    segmented = np.zeros((image.shape[0], image.shape[1]), dtype='uint8')
    res_contours = []
    for contour in contours:
        res = 1
        object_image, object_mask, matched_class, object_contour = None, None, 0, None
        for item_im, item_mask, _, _, item_class in items_info():
            item_contour, _ = cv2.findContours(to_uint8_image(item_mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            item_contour = sorted(item_contour, key=lambda cnt: cv2.contourArea(cnt))[0]
            cur_res = cv2.matchShapes(item_contour, contour, 1, 0.0)
            if cur_res < 0.3 and cur_res < res:
                res = cur_res
                matched_class = item_class
                object_contour = item_contour
                object_image = item_im
                object_mask = item_mask
        if res == 1:
            continue
        item_des = np.array([get_descriptor(Point(c[0][1], c[0][0]), object_image, (5, 5)) for c in object_contour])
        im_des = np.array([get_descriptor(Point(c[0][1], c[0][0]), image, (5, 5)) for c in contour])
        matched = match_descriptors(item_des, im_des)
        cur_segmented = _get_transformed_mask(segmented, np.array([object_contour[match[0]] for match in matched]),
                                              np.array([contour[match[1]] for match in matched]), object_mask,
                                              matched_class)
        if cur_segmented is None:
            continue
        if verbose:
            plt.imshow(cur_segmented + 220)
            plt.title(f'segmented class {matched_class}')
            plt.show()
        segmented = cur_segmented
        res_contours.append(contour)
    if verbose:
        plt.imshow(segmented + 220)
        plt.title(f'segmentation')
        plt.show()
    return segmented, res_contours


def get_items_mask(image: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, List]:
    cnt = _get_objects_contours(image, verbose)
    return _classify_objects(cnt, image, verbose)
