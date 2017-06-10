import numpy as np
import cv2


def convert_color(img, conv="RBG"):
    if conv != 'RGB':
        color_specifier = "COLOR_RGB2{}".format(conv)
        feature_image = cv2.cvtColor(img, getattr(cv2, color_specifier))
    else:
        feature_image = np.copy(img)
    return feature_image
