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
        'item1': [1, '.jpg', Area(850, 2200, 400, 1750)],
        'item2': [2, '.jpg', Area(1400, 2000, 790, 1700)],
        'item3': [3, '.jpg', Area(200, 2500, 300, 1800)],
        'item4': [4, '.jpg', Area(300, 2600, 300, 1800)],
        'item5': [5, '.jpg', Area(300, 2600, 200, 1800)],
        'item6': [6, '.jpg', Area(200, 2600, 300, 1800)],
        'item7': [7, '.jpg', Area(200, 2600, 200, 1800)],
        'item8': [8, '.jpg', Area(200, 2600, 200, 1800)],
        'item9': [9, '.jpg', Area(300, 2600, 300, 1800)],
        'item10': [10, '.jpg', Area(200, 2600, 300, 1800)],
}


def items_info(detector: Optional[cv2.SIFT] = None, mask_filter: str = '_mask'):
    data_path = package_path('data')
    for item_name, item_class in classes.items():
        item_source = read_image(f'{data_path}/{item_name}{item_class[1]}')
        item_mask = read_image(f'{data_path}/{item_name}{mask_filter}{item_class[1]}', gray=True)

        _, item_mask = cv2.threshold(item_mask, 230, 255, 0)
        kp, des = None, None
        if detector is not None:
            kp, des = detector.detectAndCompute(np.clip(item_source * 255, 0, 255).astype('uint8'), None)
        yield item_source, item_mask, kp, des, item_class[0]
