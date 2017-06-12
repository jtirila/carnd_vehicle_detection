import numpy as np
import cv2
from carnd_vehicle_detection.preprocess import color_hist, bin_spatial, get_hog_features
from carnd_vehicle_detection.preprocess import convert_color as convert_color


def find_cars(img, ystart, ystop, scale, svc, X_scaler,
              orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
              color_space, spatial_feat, hog_feat, hog_channel, hist_feat):

    work_img = np.copy(img)
    work_img = work_img.astype(np.float32) / 255

    img_tosearch = work_img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv=color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch0 = ctrans_tosearch[:, :, 0]
    ch1 = ctrans_tosearch[:, :, 1]
    ch2 = ctrans_tosearch[:, :, 2]

    # Let us put the channels into a dict for easier dynamic lookup
    channels_dict = {0: ch0, 1: ch1, 2: ch2}

    # Define blocks and steps as above
    nxblocks = (ch0.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch0.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    if hog_feat:
        common_pos_params = (orient, pix_per_cell, cell_per_block)
        common_kw_params = {'feature_vec': False}
        if hog_channel == "ALL":
            hog0 = get_hog_features(ch0, *common_pos_params, **common_kw_params)
            hog1 = get_hog_features(ch1, *common_pos_params, **common_kw_params)
            hog2 = get_hog_features(ch2, *common_pos_params, **common_kw_params)
        else:
            hog = get_hog_features(channels_dict[hog_channel], *common_pos_params, **common_kw_params)
    hot_windows = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            features = []
            if hog_feat:
                if hog_channel == "ALL":
                    hog_feat0 = hog0[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_features = np.hstack((hog_feat0, hog_feat1, hog_feat2))
                else:
                    hog_features = hog[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                features.append(hog_features)

            xleft, ytop = xpos * pix_per_cell, ypos * pix_per_cell

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
            test_prediction = svc.predict(test_features)

            # box_img = np.zeros_like(draw_img)
            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                hot_windows.append(((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
    return hot_windows
