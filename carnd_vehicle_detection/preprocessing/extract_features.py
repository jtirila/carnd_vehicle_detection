import numpy as np


def extract_features(imgs):
    """Extracts the features from the images collection
    
    :param imgs: A numpy ndarray containing the rgb images
    :return: A numpy ndarray with the transformed features 
    """

    # FIXME
    transformed = np.array([img.ravel() for img in imgs])
    return transformed

