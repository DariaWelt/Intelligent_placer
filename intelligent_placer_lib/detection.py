from typing import Optional, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.filters import gaussian, sobel
from skimage.measure import label
from skimage.segmentation import watershed

from .descriptor import get_descriptor, Point, match_descriptors
from .utils import Area, to_grayscale, to_uint8_image

MIN_OBJECT_WIDTH: int = 50
MIN_OBJECT_AREA: int = MIN_OBJECT_WIDTH * MIN_OBJECT_WIDTH
MIN_OBJECTS_DIST: int = 10


def get_polygon_mask(image: np.ndarray, area: Area) -> Optional[np.ndarray]:
    result = np.zeros(shape=(image.shape[0], image.shape[1]))
    if image.ndim == 3:
        data = rgb2gray(image)
    else:
        data = image
    im_polygon = data[:, :area.left_x]

    im_polygon = gaussian(im_polygon, 5)
    polygon_bounds = canny(im_polygon, sigma=1)
    polygons_mask = watershed(polygon_bounds)
    polygons_mask = polygons_mask > 1

    segmented_bound = np.logical_and(polygons_mask, polygon_bounds)
    bound_is_darker = np.percentile(im_polygon[segmented_bound], 25) < np.mean(im_polygon)
    bound_closed = np.percentile(im_polygon[segmented_bound], 75) < np.mean(im_polygon[polygons_mask])
    labeled, polygons_num = label(polygons_mask, connectivity=2, return_num=True)
    if not bound_closed or not bound_is_darker or polygons_num != 1:
        return None

    result[:, :area.left_x] = polygons_mask
    return result


def _get_objects_contours(image: np.ndarray) -> List:
    source = to_grayscale(gaussian(image, int(MIN_OBJECT_WIDTH / 10)))
    bounds = sobel(source)
    _, bounds = cv2.threshold(bounds, np.percentile(bounds, 95), bounds.max(),0)
    contours, hierarchy = cv2.findContours(to_uint8_image(bounds), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    im = np.zeros((bounds.shape[0], bounds.shape[1], 3))
    cv2.drawContours(im, contours, -1, (0, 255, 0))
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


def _classify_objects(contours, image: np.ndarray) -> Tuple[np.ndarray, List]:
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
        segmented = cur_segmented
        res_contours.append(contour)
    plt.imshow(segmented)
    plt.show()
    return segmented, res_contours


# def get_items_mask(image: np.ndarray, surface: np.ndarray) -> np.ndarray:
#     from .utils import items_info
#     contours = _get_objects_contours(image)
#     detector = cv2.SIFT_create(nOctaveLayers=5, edgeThreshold=30)
#     segmented = np.zeros((image.shape[0], image.shape[1]), dtype='uint8')
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         object_im = np.copy(image[y: y + h, x: x + w])
#         for item_source, item_mask, _, _, item_class in items_info(detector):
#             matched = cv2.matchTemplate(object_im, item_source, cv2.TM_CCOEFF_NORMED)
#             print(cv2.minMaxLoc(matched))
#     return segmented

def get_items_mask(image: np.ndarray, surface: np.ndarray) -> Optional[np.ndarray]:
    cnt = _get_objects_contours(image)
    return _classify_objects(cnt, image)

# def get_items_mask(image: np.ndarray, surface: np.ndarray) -> Optional[np.ndarray]:
#     from .utils import items_info
#     detector = cv2.SIFT_create(nOctaveLayers=5, edgeThreshold=30)
#     index_params = dict(algorithm=2, trees=10)
#     search_params = dict(checks=100)
#     matcher = cv2.FlannBasedMatcher(index_params, search_params)
#     segmented = np.zeros((image.shape[0], image.shape[1]))
#
#     kp, des = detector.detectAndCompute(image, None)
#     for item_source, item_mask, item_kp, item_des, item_class in items_info(detector):
#         matches = matcher.knnMatch(item_des, des, k=2)
#
#         matched = list(filter(lambda e: e[0].distance < 0.6 * e[1].distance, matches))
#
#         src_key_pts = np.float32([item_kp[m.queryIdx].pt for m, n in matched]).reshape(-1, 1, 2)
#         dst_key_pts = np.float32([kp[m.trainIdx].pt for m, n in matched]).reshape(-1, 1, 2)
#         plt.imshow(image)
#         plt.scatter(dst_key_pts[:, :, 0], dst_key_pts[:, :, 1])
#         plt.show()
#         plt.imshow(item_source)
#         plt.scatter(src_key_pts[:, :, 0], src_key_pts[:, :, 1])
#         plt.show()
#         if matched is None or len(matched) < 4 or item_kp is None:
#             continue
#         matrix, mask = cv2.findHomography(src_key_pts, dst_key_pts, cv2.RANSAC, 5.0)
#         if matrix is None:
#             continue
#         p = np.where(item_mask >= 200)
#         mask_points = [[point[1], point[0]] for point in zip(*p)]
#         new_mask_points = cv2.perspectiveTransform(np.float32(mask_points).reshape(-1, 1, 2), matrix)[:, 0, :]
#         for point in new_mask_points:
#             if segmented.shape[0] <= int(point[1]) or int(point[1]) < 0 \
#                     or segmented.shape[1] <= int(point[0]) or int(point[0]) < 0 :
#                 continue
#             #if segmented[int(point[1])][int(point[0])] != 0:
#             #    return None
#             segmented[int(point[1])][int(point[0])] = item_class
#         plt.imshow(segmented+230)
#         plt.show()
#         print(new_mask_points)
#     return segmented
