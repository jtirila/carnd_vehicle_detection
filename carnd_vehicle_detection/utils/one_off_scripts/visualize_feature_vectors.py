import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from unit_tests import TEST_IMG_DIR
import os
FULL_IMAGE = mpimg.imread(os.path.join(TEST_IMG_DIR, 'test_frame_1.png'))
SUBIMAGE_NOVEHICLE = mpimg.imread(os.path.join(TEST_IMG_DIR, 'test_subframe_1_1.png'))
SUBIMAGE_VEHICLE = mpimg.imread(os.path.join(TEST_IMG_DIR, 'test_subframe_1_2.png'))
from carnd_vehicle_detection.preprocess import extract_features, single_img_features
import pickle
from carnd_vehicle_detection import ROOT_DIR
import os

SCALER_PATH = os.path.join(ROOT_DIR, 'classifier_scaler.p')


def visualize_feature_vectors():
    """Visualize the feature vectors chosen from a couple of subimages from a frame. The selections have been picked 
    manually."""
    with open(SCALER_PATH) as scalerfile:
        scaler = pickle.load(scalerfile)
    features_novehicle = extract_features([SUBIMAGE_NOVEHICLE], color_space='YCrCb', spatial_size=(32, 32),
                                          hist_bins=32, hist_range=(0, 256), orient=9, pix_per_cell=8,
                                          cell_per_block=2, spatial_feat=True, hist_feat=True, hog_feat=True,
                                          hog_channel=0)
    scaled_features_novehicle = scaler.transform(features_novehicle)
    x = range(len(scaled_features_novehicle[0]))
    plt.plot(x, scaled_features_novehicle[0])
    plt.show()

    # features_vehicle = extract_features([SUBIMAGE_NOVEHICLE])

    # * Extract features from just thumbnail image
    # * In search for match in big picture:
    #   - Process test image in full
    #   - List just the window matching the thumbnail
    #   - Compare feature vectors
    #   - See if match is detected

if __name__ == "__main__":
    visualize_feature_vectors()
