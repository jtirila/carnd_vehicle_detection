import numpy as np
import cv2
import matplotlib.pyplot as plt
from carnd_vehicle_detection.preprocess import single_img_features, get_hog_features, convert_color, \
    extract_global_hog_features, extract_prediction_features
from carnd_vehicle_detection.traverse_image import all_windows_divisible_by


def find_cars(img, ystart_stop, xstart_stop, scale, svc, X_scaler, extract_params):

    ystart, ystop = ystart_stop
    xstart, xstop = xstart_stop
    xstart = 0 if xstart is None else xstart
    ystart = 0 if ystart is None else ystart
    xstop = img.shape[1] if xstop is None else xstop
    ystop = img.shape[0] if ystop is None else xstop

    work_img = np.copy(img)
    work_img = work_img.astype(np.float32) / 255

    img_tosearch = work_img[ystart:ystop, xstart:xstop, :]
    ctrans_tosearch = convert_color(img_tosearch, conv=extract_params['color_space'])
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch0 = ctrans_tosearch[:, :, 0]
    ch1 = ctrans_tosearch[:, :, 1]
    ch2 = ctrans_tosearch[:, :, 2]

    # Let us put the channels into a dict for easier dynamic lookup
    channels_dict = {0: ch0, 1: ch1, 2: ch2}

    # Define blocks and steps
    nxblocks = (ch0.shape[1] // extract_params['pix_per_cell']) - extract_params['cell_per_block'] + 1
    nyblocks = (ch0.shape[0] // extract_params['pix_per_cell']) - extract_params['cell_per_block'] + 1
    nfeat_per_block = extract_params['orient'] * extract_params['cell_per_block'] ** 2

    # 64 was the original sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // extract_params['pix_per_cell']) - extract_params['cell_per_block'] + 1
    cells_per_step = 1 if scale >= 2 else 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    if extract_params['hog_feat']:

        global_hog_features = extract_global_hog_features(channels_dict, extract_params['orient'],
                                                          extract_params['pix_per_cell'],
                                                          extract_params['cell_per_block'],
                                                          extract_params['hog_channel'])
    else:
        global_hog_features = None
    hot_windows = []
    window_params = {'scale': scale, 'window': window, 'ystart': ystart,
                     'ystop': ystop, 'nblocks_per_window': nblocks_per_window}
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            xleft, ytop = xpos * extract_params['pix_per_cell'], ypos * extract_params['pix_per_cell']
            window_params['xpos'] = xpos
            window_params['ypos'] = ypos
            window_params['xleft'] = xleft
            window_params['ytop'] = ytop

            params = {'global_hog_features': global_hog_features,
                      'window_params': window_params,
                      'extract_params': extract_params}
            features = extract_prediction_features(ctrans_tosearch, X_scaler, **params)
            # test_features = X_scaler.transform(np.hstack(features).reshape(1, -1))
            # plt.plot(list(range(len(features.T))), features.T)
            # plt.show()
            test_prediction = svc.predict(features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                hot_windows.append(((xbox_left + xstart, ytop_draw + ystart),
                                    (xbox_left + xstart + win_draw, ytop_draw + win_draw + ystart)))
    return hot_windows

def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    """A generator function to iterate over the windows and yield the ones for which prediction is 'vehicle', 
    otherwise (implicitly) None.
    
    :param img: the original rgb image
    :param windows: the list of windows to use for choosing candidate images
    :param clf: the classifier
    :param scaler: the scaler
    
    For the rest of the params, see the documentation of single_img_features. Some of them are used here as well 
    with the same semantics.
    
    :yield: """

    assert all_windows_divisible_by(windows, pix_per_cell, windows[0][0][0], windows[0][0][1])
    feature_image = convert_color(img, color_space)

    global_hog_features = get_global_hog_features(feature_image, orient, pix_per_cell, cell_per_block, hog_channel)

    for window in windows:
        test_img = cv2.resize(feature_image[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))

        # Determine the slices where to get the precomputed hog features.
        hog_x_start, hog_x_end, hog_y_start, hog_y_end = get_hog_extraction_coordinates(window, pix_per_cell)
        img_hog_features = get_image_hog_features(global_hog_features, hog_x_start, hog_x_end,
                                                  hog_y_start, hog_y_end, hog_channel)
        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       bins_range=hist_range,
                                       img_hog_features=img_hog_features,
                                       spatial_feat=spatial_feat, hog_channel=hog_channel,
                                       hist_feat=hist_feat, hog_feat=hog_feat)

        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))

        # 6) Predict using your classifier
        prediction = clf.predict(test_features)

        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            yield window


def get_global_hog_features(img, orient, pix_per_cell, cell_per_block, hog_channel):
    if hog_channel == 'ALL':
        global_hog_features = []
        for channel in range(img.shape[2]):
            global_hog_features.append(
                get_hog_features(img[:, :, channel],
                                 orient, pix_per_cell, cell_per_block,
                                 vis=False, feature_vec=False))
    else:
        global_hog_features = get_hog_features(img[:, :, hog_channel], orient,
                                               pix_per_cell, cell_per_block, vis=False, feature_vec=False)
    return global_hog_features

def get_image_hog_features(global_hog_features, hog_x_start, hog_x_end, hog_y_start, hog_y_end, hog_channel):
    if hog_channel == "ALL":
        img_hog_features = [global_hog_features[0][hog_y_start:hog_y_end, hog_x_start:hog_x_end, :, :, :],
                            global_hog_features[1][hog_y_start:hog_y_end, hog_x_start:hog_x_end, :, :, :],
                            global_hog_features[2][hog_y_start:hog_y_end, hog_x_start:hog_x_end, :, :, :]
                            ]
    else:
        img_hog_features = global_hog_features[hog_y_start:hog_y_end, hog_x_start:hog_x_end, :, :, :]
    return img_hog_features


def get_hog_extraction_coordinates(window, pix_per_cell):
    hog_x_start = int(window[0][0] / pix_per_cell)
    hog_x_end = int(window[1][0] / pix_per_cell) - 1
    hog_y_start = int(window[0][1] / pix_per_cell)
    hog_y_end = int(window[1][1] / pix_per_cell) - 1
    return hog_x_start, hog_x_end, hog_y_start, hog_y_end
