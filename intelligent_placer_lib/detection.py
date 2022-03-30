from typing import List, Tuple, Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from skimage.filters import sobel

from .descriptor import get_descriptor, Point, match_descriptors
from .utils import to_grayscale, to_uint8_image, get_item, RED_CIRCLE_CLASS, BLUE_RECT_CLASS

MIN_OBJECT_WIDTH: int = 50
MIN_DIST_BETWEEN: int = 60
MIN_OBJECT_AREA: int = MIN_OBJECT_WIDTH * MIN_OBJECT_WIDTH


def _get_objects_contours(image: np.ndarray, verbose: bool = False) -> Tuple[Any, List]:
    source = to_grayscale(image)
    bounds = sobel(source)
    _, bounds = cv2.threshold(bounds, np.percentile(bounds, 98), bounds.max(), 0)
    bounds = cv2.morphologyEx(bounds, cv2.MORPH_CLOSE, (MIN_DIST_BETWEEN, MIN_DIST_BETWEEN))
    contours, hierarchy = cv2.findContours(to_uint8_image(bounds), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    good_cnts = list(filter(lambda contour: cv2.contourArea(contour) >= MIN_OBJECT_AREA, contours))
    if verbose:
        im = image.copy()
        cv2.drawContours(im, good_cnts, -1, (0, 255, 0), 5)
        plt.imshow(im)
        plt.title('image contours')
        plt.show()
    poly_index = np.argmin([int(cv2.moments(cnt)['m10'] / cv2.moments(cnt)['m00']) for cnt in good_cnts])
    if verbose:
        im = image.copy()
        cv2.drawContours(im, [good_cnts[poly_index]], -1, (0, 0, 255), 5)
        plt.imshow(im)
        plt.title('polygon contour')
        plt.show()
    return good_cnts[poly_index], good_cnts[:poly_index]+good_cnts[poly_index+1:]


def _get_contour_mask(contour, shape):
    mask = np.zeros(shape)
    return cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)


def get_channels_average(object_mask: np.ndarray, object_image: np.ndarray) -> np.ndarray:
    object_part = np.array([object_image[..., i] * (object_mask / 255).astype('uint8') for i in range(object_image.shape[-1])])
    return np.sum(np.sum(object_part, axis=1), axis=1) / np.sum(object_mask > 0) / 255


def is_red_circle(object_mask: np.ndarray, object_image: np.ndarray, object_contour, circle_contour) -> bool:
    channels_av = get_channels_average(object_mask, object_image)
    is_red = not np.isnan(channels_av[0]) and channels_av[0] > 0.6 and channels_av[1] < 0.3 and channels_av[2] < 0.3
    if not is_red:
        return False
    cur_res = cv2.matchShapes(object_contour, circle_contour, cv2.cv2.CONTOURS_MATCH_I3, 0.0)
    return cur_res < 0.1


def is_blue_rectangle(object_mask: np.ndarray, object_image: np.ndarray, object_contour) -> bool:
    channels_av = get_channels_average(object_mask, object_image)
    is_blue = not np.isnan(channels_av[2]) and channels_av[2] > 0.6 and channels_av[0] < 0.2 and channels_av[1] < 0.6
    if not is_blue:
        return False
    outer_rect = np.int0(cv2.boxPoints(cv2.minAreaRect(object_contour)))
    object_area, rect_area = cv2.contourArea(object_contour), cv2.contourArea(outer_rect)
    return object_area / rect_area > 0.9


def _add_segmented_object(segmentation: np.ndarray, contour, object_class: int, verbose: bool = False) -> np.ndarray:
    cur_segmented = np.zeros(segmentation.shape)
    cv2.drawContours(cur_segmented, [contour], -1, object_class, cv2.FILLED)
    cv2.drawContours(segmentation, [contour], -1, object_class, cv2.FILLED)
    if verbose:
        plt.imshow(cur_segmented + 220)
        plt.title(f'segmented class {object_class}')
        plt.show()
    return segmentation


def _classify_objects(contours, polygon_max_x: int, image: np.ndarray, verbose: bool = False) -> np.ndarray:
    from .utils import items_info

    segmented = np.zeros((image.shape[0], image.shape[1]), dtype='uint8')
    res_contours = []
    for i, contour in enumerate(contours):
        if min([point[0][0] for point in contour]) < polygon_max_x:
            return np.zeros((image.shape[0], image.shape[1]), dtype='uint8')
        contour_mask = _get_contour_mask(contour, segmented.shape)
        x, y, w, h = cv2.boundingRect(contour)
        _, _, _, circle_cnt = get_item(RED_CIRCLE_CLASS-1)
        if is_red_circle(contour_mask[y: y + h, x: x + w], image[y: y + h, x: x + w], contour, circle_cnt):
            segmented = _add_segmented_object(segmented, contour, RED_CIRCLE_CLASS, verbose)
        elif is_blue_rectangle(contour_mask[y: y + h, x: x + w], image[y: y + h, x: x + w], contour):
            segmented = _add_segmented_object(segmented, contour, BLUE_RECT_CLASS, verbose)
        else:
            im_des = np.array([get_descriptor(Point(c[0][1], c[0][0]), image, (5, 5)) for c in contour])
            res = 1
            matched_class, best_matched = 0, 0
            for item_im, item_mask, item_contour, item_class, _, _ in items_info():
                if item_class in [RED_CIRCLE_CLASS, BLUE_RECT_CLASS]:
                    continue
                cur_res = cv2.matchShapes(item_contour, contour, cv2.cv2.CONTOURS_MATCH_I3, 0.0)
                if cur_res < 0.4:
                    item_des = np.array(
                        [get_descriptor(Point(c[0][1], c[0][0]), item_im, (5, 5)) for c in item_contour])
                    pairs = match_descriptors(item_des, im_des)
                    distances = [distance.euclidean(item_des[match[0]], im_des[match[1]]) /
                                 (np.linalg.norm(item_des[match[0]]) * np.linalg.norm(im_des[match[1]]))
                                 for match in pairs]
                    cur_matched = len(list(filter(lambda elem: elem < 0.002, distances)))
                    if cur_matched > best_matched:
                        best_matched = cur_matched
                        res = cur_res
                        matched_class = item_class
            if res == 1:
                continue
            if best_matched < 10:
                continue
            segmented = _add_segmented_object(segmented, contour, matched_class, verbose)
        res_contours.append(contour)

    if verbose:
        plt.imshow(segmented + 220)
        plt.title(f'segmentation')
        plt.show()
    return segmented


def get_items_mask(image: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    poly_cnt, obj_cnts = _get_objects_contours(image, verbose)
    maxx = max([point[0][0] for point in poly_cnt])
    return poly_cnt, _classify_objects(obj_cnts, maxx, image, verbose)
