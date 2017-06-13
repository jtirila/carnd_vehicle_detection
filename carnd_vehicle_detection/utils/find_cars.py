import numpy as np
import cv2
from carnd_vehicle_detection.preprocess import color_hist, bin_spatial, get_hog_features, extract_global_hog_features
from carnd_vehicle_detection.preprocess import convert_color as convert_color
from carnd_vehicle_detection.preprocess.extract_features import extract_prediction_features


def find_cars(img, ystart, ystop, scale, svc, X_scaler, extract_params):

    work_img = np.copy(img)
    work_img = work_img.astype(np.float32) / 255

    img_tosearch = work_img[ystart:ystop, :, :]
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
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    if extract_params['hog_feat']:

        global_hog_features = extract_global_hog_features(channels_dict, extract_params['orient'],
                                                         extract_params['pix_per_cell'], extract_params['cell_per_block'],
                                                         extract_params['hog_channel'])
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
            test_features = X_scaler.transform(np.hstack(features).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                hot_windows.append(((xbox_left, ytop_draw + ystart),
                                    (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
    return hot_windows
