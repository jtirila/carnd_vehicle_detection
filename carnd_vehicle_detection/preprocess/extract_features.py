import numpy as np
from carnd_vehicle_detection.preprocess import get_hog_features, color_hist, bin_spatial, convert_color
import matplotlib.pyplot as plt
import cv2

WINDOWS_PARAMS_KEYS = {'nblocks_per_window', 'window', 'ypos', 'xpos', 'ystart', 'ystop', 'scale', 'xleft', 'ytop'}
HOG_PARAMS_KEYS = {'hog0', 'hog1', 'hog2'}
EXTRACT_PARAMS_KEYS = {'color_space', 'orient', 'pix_per_cell', 'cell_per_block', 'hog_channel',
                       'spatial_size', 'hist_bins', 'spatial_feat', 'hist_feat', 'hog_feat'}


def extract_prediction_features(ctrans_tosearch, X_scaler, **params):
    """Extract features from a subimage of a frame, based on the previously extracted global HOG features and
     other parameter values.
     
     :param ctrans_tosearch: FIXME
     :param X_scaler: FIXME
     :param params: A dict with the structure outlined in the assertiond ans assignments below.
    
    :return: A list of 'hot windows', in the format 
                 [((x_topleft_ho1, y_topleft_hot1), (x_bottomright_hot1, y_bottomright_hot1)), 
                 ((x_topleft_hot2, y_topleft_hot2), (x_bottomright_hot2, y_bottomright_hot2)), ...]"""

    global_hog_features = params['global_hog_features']
    window_params = params['window_params']
    extract_params = params['extract_params']

    assert WINDOWS_PARAMS_KEYS == set(window_params.keys())
    nblocks_per_window = window_params['nblocks_per_window']
    window = window_params['window']
    ypos = window_params['ypos']
    xpos = window_params['xpos']
    ystart = window_params['ystart']
    ystop = window_params['ystop']
    xleft = window_params['xleft']
    ytop = window_params['ytop']

    assert EXTRACT_PARAMS_KEYS == set(extract_params.keys())

    color_space = extract_params['color_space']
    orient = extract_params['orient']
    pix_per_cell = extract_params['pix_per_cell']
    cell_per_block = extract_params['cell_per_block']
    hog_channel = extract_params['hog_channel']
    spatial_size = extract_params['spatial_size']
    hist_bins = extract_params['hist_bins']
    spatial_feat = extract_params['spatial_feat']
    hist_feat = extract_params['hist_feat']
    hog_feat = extract_params['hog_feat']

    features = []
    if extract_params['hog_feat']:
        if set(global_hog_features.keys()) == {'hog'}:
            hog = global_hog_features['hog']
            hog0 = hog1 = hog2 = None
        else:
            assert HOG_PARAMS_KEYS == set(global_hog_features.keys())
            hog0 = global_hog_features['hog0']
            hog1 = global_hog_features['hog1']
            hog2 = global_hog_features['hog2']
            hog = None



        if extract_params['hog_channel'] == "ALL":
            hog_feat0 = hog0[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat0, hog_feat1, hog_feat2))
        else:
            hog_features = hog[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
        features.append(hog_features)


    # Extract the image patch
    subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

    if spatial_feat:
        # Get color features
        spatial_features = bin_spatial(subimg, size=spatial_size)
        features.append(spatial_features)
    if hist_feat:
        hist_features = color_hist(subimg, nbins=hist_bins)
        features.append(hist_features)

    # Scale features and make a prediction
    test_features = X_scaler.transform(np.hstack(features).reshape(1, -1))
    return test_features


def single_img_features(img, color_space='RGB', spatial_size=(32, 32), hist_bins=32, bins_range=(0, 256),
                        img_hog_features=None, spatial_feat=True, hist_feat=True, hog_feat=True, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel="ALL"):

    """A function to extract the features from an image.
    
    :param img: the original rgb image
    :param color_space: the color space to convert the image into
    :param spatial_size: For the bin_spatial method, the image size to use
    :param hist_bins: for the color histogram the number of bins to use
    :param bins_range: the value range for color histogram
    :param img_hog_features: for single-pass hog processing, the hog features corresponding to this particular image 
           segment
    :param orient: The number of orientation bins for HOG 
    :param pix_per_cell: cell size for HOG
    :param cell_per_block: block size for HOG
    :param hog_channel: channel for HOG
    :param spatial_feat: Boolean, whether to use bin_spatial features in the final feature vector
    :param hist_feat: Boolean, whether to use color histogram features in the final feature vector
    :param hog_feat: Boolean, whether to use HOG features in the final feature vector"""

    # 1) Define an empty list to receive features
    img_features = []

    # 2) Convert color space (if specified as other than RGB)
    img_copy = convert_color(img, color_space)

    # 7) Compute HOG features if flag is set
    if hog_feat:
        hog_features = []
        if hog_channel == "ALL":
            for channel in range(img_copy.shape[2]):
                hog_features.append(
                    get_hog_features(img_copy[:, :, channel], orient, pix_per_cell,
                                     cell_per_block, vis=False, feature_vec=False)
                )
        else:
            hog_features = get_hog_features(img_copy[:, :, hog_channel], orient, pix_per_cell,
                                            cell_per_block, vis=False, feature_vec=False)
        img_features.append(np.ravel(hog_features))

    # 3) Compute spatial features if flag is set

    if spatial_feat:
        spatial_features = bin_spatial(img_copy, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)

    # 5) Compute histogram features if flag is set
    if hist_feat:
        hist_features = color_hist(img_copy, nbins=hist_bins, bins_range=bins_range)
        # 6) Append features to list
        img_features.append(hist_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)


def extract_global_hog_features(channels_dict, orient, pix_per_cell, cell_per_block, hog_channel):
    common_pos_params = (orient, pix_per_cell, cell_per_block)
    common_kw_params = {'feature_vec': False}
    if hog_channel == "ALL":
        hog0 = get_hog_features(channels_dict[0], *common_pos_params, **common_kw_params)
        hog1 = get_hog_features(channels_dict[1], *common_pos_params, **common_kw_params)
        hog2 = get_hog_features(channels_dict[2], *common_pos_params, **common_kw_params)
        return {'hog0': hog0, 'hog1': hog1, 'hog2': hog2}
    else:
        return {'hog': get_hog_features(channels_dict[hog_channel],
                                        *common_pos_params,
                                        **common_kw_params)}


