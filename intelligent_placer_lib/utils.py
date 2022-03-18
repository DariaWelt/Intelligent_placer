from typing import NamedTuple, Optional

import cv2
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.filters import gaussian

IMAGE_POSTFIXES = ['.jpg', '.tif', '.png', '.tiff']


class Area(NamedTuple):
    left_x: int
    right_x: int
    up_y: int
    down_y: int


def read_image(image_path: str, gray: bool = False) -> np.ndarray:
    image = np.array(Image.open(image_path))
    if image.ndim > 3:
        raise ValueError('more than one image recieved. Only 3 dim arrays are supported')
    channels_num = min(image.shape)
    if image.ndim > 1 and channels_num > 3:
        raise ValueError(f'Image contains {channels_num} channels. Only rgb or grayscale images are supported')
    channels_index = image.shape.index(channels_num)
    if image.ndim > 1 and channels_index != image.ndim - 1:
        image = np.moveaxis(image, channels_index, -1)
    if gray:
        image = to_grayscale(image)
    return to_uint8_image(image)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    image = rgb2gray(image)
    return to_uint8_image(image)


def to_uint8_image(image: np.ndarray) -> np.ndarray:
    image = image / image.max()
    image = np.clip(image * 255, 0, 255).astype('uint8')
    return image


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
    for item_name, item_class in classes.items():
        item_source = read_image(f'../intelligent_placer_lib/data/{item_name}{item_class[1]}')
        item_mask = read_image(f'../intelligent_placer_lib/data/{item_name}{mask_filter}{item_class[1]}', gray=True)

        item_source = item_source[item_class[2].up_y: item_class[2].down_y, item_class[2].left_x: item_class[2].right_x]
        item_mask = item_mask[item_class[2].up_y: item_class[2].down_y, item_class[2].left_x: item_class[2].right_x]

        _, item_mask = cv2.threshold(item_mask, 230, 255, 0)
        kp, des = None, None
        if detector is not None:
            kp, des = detector.detectAndCompute(np.clip(item_source * 255, 0, 255).astype('uint8'), None)
        yield item_source, item_mask, kp, des, item_class[0]
