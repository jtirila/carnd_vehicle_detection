import numpy as np
import cv2


def bin_spatial(img, size=(32, 32)):
    """Just resize an image and return the color values as a 1-dimensional vector
    
    :param img: the original image
    :param size: A two-tuple containing the new size
    :return: a 1-dim feature vector"""

    img_cc = np.copy(img)
    small_img = cv2.resize(img_cc, size)
    features = small_img.ravel()
    # Return the feature vector
    return features
