import numpy as np
import cv2
from carnd_vehicle_detection.preprocess import single_img_features, get_hog_features, convert_color
from carnd_vehicle_detection.traverse_image import all_windows_divisible_by


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
        hog_channel=hog_channel, spatial_feat=True,
        hist_feat=True, hog_feat=True) if on_window is not None]


def fixme_something(img, windows, clf, scaler, color_space='RGB',
                    spatial_size=(32, 32), hist_bins=32,
                    hist_range=(0, 256), orient=9,
                    pix_per_cell=8, cell_per_block=2,
                    hog_channel=0, spatial_feat=True,
                    hist_feat=True, hog_feat=True):

    train_x_len = 64
    assert all_windows_divisible_by(windows, pix_per_cell, windows[0][0][0], windows[0][0][1])
    feature_image = convert_color(img, color_space)

    # Calculate the scaled pixels per cell count to make hog transform comparable to training results
    # on this scale as well
    new_pix_per_cell = int((((windows[0][1][0] - windows[0][0][0])) / train_x_len) * pix_per_cell)

    if hog_channel == 'ALL':
        global_hog_features = []
        for channel in range(feature_image.shape[2]):
            global_hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                 orient, new_pix_per_cell, cell_per_block,
                                                 vis=False, feature_vec=False))
        global_hog_features = np.dstack(global_hog_features)
    else:
        global_hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                        new_pix_per_cell, cell_per_block, vis=False, feature_vec=False)

    for window in windows:
        test_img = cv2.resize(feature_image[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        hog_x_start = int(window[0][0] / new_pix_per_cell)
        hog_x_end = int(window[1][0] / new_pix_per_cell) - 1
        hog_y_start = int(window[0][1] / new_pix_per_cell)
        hog_y_end = int(window[1][1] / new_pix_per_cell) - 1
        img_hog_features = global_hog_features[hog_y_start:hog_y_end, hog_x_start:hog_x_end, :, :, :]
        img_hog_features = img_hog_features.ravel()
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

