import numpy as np
import cv2


def convert_color(img, conv="RBG"):
    """Convert color to the specified color space.
    
    :param img: The original image, assumed to be in RGB
    :param conv: A string containing the new colorspace name, accepted strings are HSV, HLS; YCrCb, FIXME what else
    
    :return: the transformed image"""
    if conv != 'RGB':
        color_specifier = "COLOR_RGB2{}".format(conv)
        feature_image = cv2.cvtColor(img, getattr(cv2, color_specifier))
    else:
        feature_image = np.copy(img)
    return feature_image


def normalize_luminosity(img):
    """Normalize contrast as per http://stackoverflow.com/a/38312281
    
    :param img: An image in RBG format"""
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    # equalize the histogram of the Y channel
    channel = img_yuv[:, :, 0]
    # plt.imshow(channel)
    # plt.show()
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    color_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    hls_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2HLS)
    hls_image[:, :, 2] = cv2.equalizeHist(hls_image[:, :, 2])
    color_image = cv2.cvtColor(hls_image, cv2.COLOR_HLS2RGB)
    # color_image[s_channel < 100] = 30
    return color_image

