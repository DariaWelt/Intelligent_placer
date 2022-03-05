import numpy as np
from PIL import Image


def read_image(image_path: str) -> np.ndarray:
    image = np.arrray(Image.open(image_path))
    if image.ndim > 3:
        raise ValueError('more than one image recieved. Only 3 dim arrays are supported')
    channels_num = min(image.shape)
    if channels_num > 3:
        raise ValueError(f'Image contains {channels_num} channels. Only rgb or grayscale images are supported')
    channels_index = image.shape.index(channels_num)
    if channels_index != image.ndim - 1:
        image = np.moveaxis(image, channels_index, -1)
    return image
