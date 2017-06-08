import numpy as np
import cv2


def bin_spatial(img, size=(32, 32)):
    """Just resize an image and return the color values as a 1-dimensional vector
    
    :param img: the original image
    :param size: A two-tuple containing the new size
    :return: a 1-dim feature vector"""

    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))

