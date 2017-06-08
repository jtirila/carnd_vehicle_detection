import numpy as np
import cv2
from carnd_vehicle_detection.preprocess import single_img_features, get_hog_features, color_convert


def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    # 1) Create an empty list to receive positive detection windows
    return [on_window for on_window in fixme_something(
        img, windows, clf, scaler, color_space='RGB',
        spatial_size=(32, 32), hist_bins=32,
        hist_range=(0, 256), orient=9,
        pix_per_cell=8, cell_per_block=2,
        hog_channel=0, spatial_feat=True,
        hist_feat=True, hog_feat=True) if on_window is not None]


def fixme_something(img, windows, clf, scaler, color_space='RGB',
                    spatial_size=(32, 32), hist_bins=32,
                    hist_range=(0, 256), orient=9,
                    pix_per_cell=8, cell_per_block=2,
                    hog_channel=0, spatial_feat=True,
                    hist_feat=True, hog_feat=True):

    feature_image = color_convert(img, color_space)
    if hog_channel == 'ALL':
        global_hog_features = []
        for channel in range(feature_image.shape[2]):
            global_hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                 orient, pix_per_cell, cell_per_block,
                                                 vis=False, feature_vec=False))
    else:
        global_hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                        pix_per_cell, cell_per_block, vis=False, feature_vec=False)

    for window in windows:
        test_img = cv2.resize(feature_image[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        x_start_ind, x_end_ind = 1, 10
        y_start_ind, y_end_ind = 1, 10
        img_hog_features = global_hog_features[y_start_ind:y_end_ind, x_start_ind:x_end_ind, :, :, :]
        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       img_hog_features=img_hog_features,
                                       spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)

        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
           yield window

