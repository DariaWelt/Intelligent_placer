import os
from typing import NamedTuple, Optional

import cv2
import numpy as np
from PIL import Image
from skimage.color import rgb2gray

IMAGE_POSTFIXES = ['.jpg', '.tif', '.png', '.tiff']


class Area(NamedTuple):
    left_x: int
    right_x: int
    up_y: int
    down_y: int


IM_LARGER_SIDE = 1000
RED_CIRCLE_CLASS = 4
BLUE_RECT_CLASS = 1


def read_image(image_path: str, gray: bool = False) -> np.ndarray:
    image = np.array(Image.open(image_path))
    if image.ndim > 3:
        raise ValueError('more than one image received. Only 3 dim arrays are supported')
    channels_num = min(image.shape)
    if image.ndim > 1 and channels_num > 3:
        raise ValueError(f'Image contains {channels_num} channels. Only rgb or grayscale images are supported')
    channels_index = image.shape.index(channels_num)
    if image.ndim > 1 and channels_index != image.ndim - 1:
        image = np.moveaxis(image, channels_index, -1)
    resize_scale = IM_LARGER_SIDE / max(image.shape)
    image = cv2.resize(image, (int(image.shape[1] * resize_scale), int(image.shape[0] * resize_scale)),
                       interpolation=cv2.INTER_CUBIC)
    if gray:
        image = to_grayscale(image)
    return to_uint8_image(image)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    image = rgb2gray(image)
    return to_uint8_image(image)


def to_uint8_image(image: np.ndarray) -> np.ndarray:
    m = image.max()
    image = image / m if m != 0 else image * 0
    image = np.clip(image * 255, 0, 255).astype('uint8')
    return image


def package_path(*paths, package_directory=os.path.dirname(os.path.abspath(__file__))):
    return os.path.join(package_directory, *paths)


classes = {
        'item1': [1, '.jpg'],
        'item2': [2, '.jpg'],
        'item3': [3, '.jpg'],
        'item4': [4, '.jpg'],
        'item5': [5, '.jpg'],
        'item6': [6, '.jpg'],
        'item7': [7, '.jpg'],
        'item8': [8, '.jpg'],
        'item9': [9, '.jpg'],
        'item10': [10, '.jpg'],
}


def load_items(mask_filter: str = '_mask'):
    data_path = package_path('data')
    items = []
    for item_name, item_info in classes.items():
        item_source = read_image(f'{data_path}/{item_name}{item_info[1]}')
        item_mask = read_image(f'{data_path}/{item_name}{mask_filter}{item_info[1]}', gray=True)
        _, item_mask = cv2.threshold(item_mask, 230, 255, 0)
        item_contour, _ = cv2.findContours(to_uint8_image(item_mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        item_contour = sorted(item_contour, key=lambda cnt: cv2.contourArea(cnt))[0]
        items.append([item_info[0], item_source, item_mask, item_contour])
    items.sort(key=lambda x: x[0])
    return items


items = load_items()


def get_item(i: int):
    return items[i]


def items_info(detector: Optional[cv2.SIFT] = None, mask_filter: str = '_mask'):
    for i, item_info in enumerate(items):
        kp, des = None, None
        if detector is not None:
            kp, des = detector.detectAndCompute(np.clip(item_info[1] * 255, 0, 255).astype('uint8'), None)
        yield item_info[1], item_info[2], item_info[3], item_info[0], kp, des
