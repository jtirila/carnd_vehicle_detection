import cv2


def bin_spatial(img, color_space='RGB', size=(32, 32)):
    """FIXME: document, add entries to the list of allowed color spaces"""
    if not color_space == 'RGB':
        try:
            assert color_space in ("HLS", "HSV", "LUV", "GRAY")
        except AssertionError:
            pass

        colorspace_val = getattr(cv2, "COLOR_RGB2{}".format(color_space))
        # Convert image to new color space (if specified)
        # Use cv2.resize().ravel() to create the feature vector
        img_cc = cv2.cvtColor(img, colorspace_val)
    else:
        img_cc = img

    small_img = cv2.resize(img_cc, size)
    features = small_img.ravel()  # Remove this line!
    # Return the feature vector
    return features
