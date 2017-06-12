import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from unit_tests import TEST_IMG_DIR
from copy import deepcopy
import os
from carnd_vehicle_detection import EXTRACT_PARAMS
from carnd_vehicle_detection.preprocess import single_img_features
from carnd_vehicle_detection.visualize import one_by_two_plot
from carnd_vehicle_detection.traverse_image import get_global_hog_features, get_image_hog_features, \
    get_hog_extraction_coordinates
import pickle
from carnd_vehicle_detection import ROOT_DIR
import os
import cv2


def _discard_alpha_channel(img):
    return img[:, :, :3]

FULL_IMAGE = _discard_alpha_channel(mpimg.imread(os.path.join(TEST_IMG_DIR, 'test_frame_1.png')))
SUBIMAGE_SKY_64_64 = _discard_alpha_channel(mpimg.imread(os.path.join(TEST_IMG_DIR, 'test_subframe_1_1.png')))
SUBIMAGE_ROAD_64_64 = _discard_alpha_channel(mpimg.imread(os.path.join(TEST_IMG_DIR, 'test_subframe_1_3.png')))
SUBIMAGE_ROAD_2_64_64 = _discard_alpha_channel(mpimg.imread(os.path.join(TEST_IMG_DIR, 'test_subframe_1_4.png')))
SUBIMAGE_ROAD_SCALED = _discard_alpha_channel(mpimg.imread(os.path.join(TEST_IMG_DIR, 'test_subframe_1_5.png')))
SUBIMAGE_ROAD_SCALED_2 = _discard_alpha_channel(mpimg.imread(os.path.join(TEST_IMG_DIR, 'test_subframe_1_6.png')))
SCALER_PATH = os.path.join(ROOT_DIR, 'classifier_scaler.p')
with open(SCALER_PATH, 'rb') as scalerfile:
    SCALER = pickle.load(scalerfile)


def visualize_feature_vectors():
    """Visualize the feature vectors chosen from a couple of subimages from a frame. The selections have been picked 
    manually."""

    # Extract and scale the params for non-scaled windows
    features_sky = SCALER.transform(np.ravel(single_img_features(SUBIMAGE_SKY_64_64, **EXTRACT_PARAMS)))
    features_road = SCALER.transform(np.ravel(single_img_features(SUBIMAGE_ROAD_64_64, **EXTRACT_PARAMS)))
    features_road_2 = SCALER.transform(np.ravel(single_img_features(SUBIMAGE_ROAD_2_64_64, **EXTRACT_PARAMS)))

    # Extract and scale params for 1.5 scaled windows
    extract_params = deepcopy(EXTRACT_PARAMS)
    extract_params['pix_per_cell'] = 12
    features_road_scaled = SCALER.transform(np.ravel(single_img_features(SUBIMAGE_ROAD_SCALED, **extract_params)))
    features_road_scaled_2 = SCALER.transform(np.ravel(single_img_features(SUBIMAGE_ROAD_SCALED_2, **extract_params)))

    plt.imshow(FULL_IMAGE)
    plt.show()
    one_by_two_plot(SUBIMAGE_SKY_64_64, SUBIMAGE_ROAD_64_64, None, None, "sky", "Road")

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.set_title("Non-vehicle, scaled features")
    ax1.plot(list(range(len(features_sky))), features_sky)
    ax2.set_title("Vehicle, scaled features")
    ax2.plot(list(range(len(features_road))), features_road)
    plt.show()

    wins_per_scale = {1: [((128, 400), (192, 464)), ((248, 400), (312, 464))], 1.5: [((120, 396), (216, 492)), ((144, 420), (240, 516))]}
    ref_feat_per_scale = {1: [features_road, features_road_2], 1.5: [features_road_scaled, features_road_scaled_2]}
    for scale in (1, 1.5):
        scale_pix_per_cell = int(scale * EXTRACT_PARAMS['pix_per_cell'])
        windows = wins_per_scale[scale]

        hog_x_start, hog_x_end, hog_y_start, hog_y_end = get_hog_extraction_coordinates(
            windows[0], scale_pix_per_cell)
        hog_x_start_2, hog_x_end_2, hog_y_start_2, hog_y_end_2 = get_hog_extraction_coordinates(
            windows[1], scale_pix_per_cell)

        global_hog_features = get_global_hog_features(FULL_IMAGE, pix_per_cell=scale_pix_per_cell,
                                                      cell_per_block=EXTRACT_PARAMS['cell_per_block'],
                                                      hog_channel=EXTRACT_PARAMS['hog_channel'],
                                                      orient=EXTRACT_PARAMS['orient'])

        img_hog_features = get_image_hog_features(global_hog_features, hog_x_start, hog_x_end, hog_y_start, hog_y_end,
                                                  EXTRACT_PARAMS['hog_channel'])

        img_hog_features_2 = get_image_hog_features(global_hog_features, hog_x_start_2, hog_x_end_2, hog_y_start_2, hog_y_end_2,
                                                    EXTRACT_PARAMS['hog_channel'])

        extract_params = deepcopy(EXTRACT_PARAMS)
        extract_params['img_hog_features'] = img_hog_features
        subfeatures_vehicle = SCALER.transform(np.ravel(single_img_features(SUBIMAGE_ROAD_64_64, **extract_params)))
        # scaled_subfeatures_vehicle = SCALER.transform(subfeatures_vehicle)
        x_subfeatures_novehicle = list(range(len(subfeatures_vehicle)))
        f2, (ax21, ax22) = plt.subplots(1, 2, figsize=(20, 10))
        ax21.set_title("Single-pass HOG extraction results")
        ax21.plot(x_subfeatures_novehicle, subfeatures_vehicle)
        ax22.set_title("Per-image HOG extraction results")
        ax22.plot(list(range(len(ref_feat_per_scale[scale][0]))), ref_feat_per_scale[scale][0])
        f2.suptitle("First window for scale {}".format(scale))
        plt.show()

        # plt.plot(list(range(81)), features_vehicle[120:201] - subfeatures_vehicle[120:201])
        plt.plot(list(range(len(features_road))), ref_feat_per_scale[scale][0] - subfeatures_vehicle)
        plt.show()


        extract_params['img_hog_features'] = img_hog_features_2
        subfeatures_vehicle_2 = SCALER.transform(np.ravel(single_img_features(SUBIMAGE_ROAD_64_64, **extract_params)))
        f3, (ax31, ax32) = plt.subplots(1, 2, figsize=(20, 10))
        ax31.set_title("Single-pass HOG extraction results")
        ax31.plot(x_subfeatures_novehicle, subfeatures_vehicle_2)
        ax32.set_title("Per-image HOG extraction results")
        ax32.plot(list(range(len(ref_feat_per_scale[scale][1]))), ref_feat_per_scale[scale][1])
        f2.suptitle("Second window for scale {}".format(scale))
        plt.show()

        plt.plot(list(range(len(features_road_2))), ref_feat_per_scale[scale][1] - subfeatures_vehicle_2)
        # plt.plot(list(range(81)), ref_feat_per_scale[scale][1][120:201] - subfeatures_vehicle_2[120:201])
        plt.show()

        small_img = cv2.resize(SUBIMAGE_ROAD_SCALED_2, (64, 64))
        subfeatures_from_64_64 = SCALER.transform(np.ravel(single_img_features(small_img, **EXTRACT_PARAMS)))

    f3, (ax31, ax32) = plt.subplots(1, 2, figsize=(20, 10))
    ax31.set_title("Final image, Single-pass HOG extraction results")
    ax31.plot(list(range(len(features_road_2))), features_road_2)
    ax32.set_title("Final image, Per-image 64x64 features")
    ax32.plot(list(range(len(subfeatures_from_64_64))), subfeatures_from_64_64)
    plt.show()

    plt.plot(list(range(len(features_road_2))), subfeatures_from_64_64 - subfeatures_vehicle_2)
    # plt.plot(list(range(81)), ref_feat_per_scale[scale][1][120:201] - subfeatures_vehicle_2[120:201])
    plt.show()

if __name__ == "__main__":
    visualize_feature_vectors()
