import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import cv2
from carnd_vehicle_detection import ROOT_DIR
import os
from carnd_vehicle_detection.preprocess import normalize_luminosity, convert_color

COUNTER = 0

def find_cars(img, ystart, ystop, xstart, xstop, scale, svc, X_scaler, color_space="RGB", hog_channel="ALL", orient=9, pix_per_cell=8, cell_per_block=2, spatial_size=(32, 32), hist_bins=32, hog_feat=True, hist_feat=True, spatial_feat=True):
    draw_img = np.copy(img)
    # draw_img = draw_img.astype(np.float32) / 255
    global COUNTER
    COUNTER += 1

    # print("COUNTER: {}, scale: {}, color_space: {}".format(COUNTER, scale, color_space))
    ystart = 0 if ystart is None else ystart
    ystop = draw_img.shape[0] if ystop is None else ystop
    xstart = 0 if xstart is None else xstart
    xstop = draw_img.shape[1] if xstop is None else xstop

    img_tosearch = draw_img[ystart:ystop, xstart:xstop, :]
    ctrans_tosearch = convert_color(img_tosearch, conv=color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    else:
        pass
        # if COUNTER == 7:
        #     plt.imshow(ctrans_tosearch)
        #     plt.show()


    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    # Jump around more for smaller scales, we don't need to reliable detections far away.
    if scale < 1:
        cells_per_step = 3  # Instead of overlap, define how many cells to step
    elif scale < 2.1:
        cells_per_step = 2
    else:
        cells_per_step = 1

    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    hot_windows = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            raw_features = np.hstack((hog_features, spatial_features, hist_features)).reshape(1, -1)
            test_features = X_scaler.transform(raw_features)
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)
            # if COUNTER == 7:
            #     plt.imshow(subimg.astype(np.uint8) * 255)
            #     plt.title("Predicdtion result: {}".format(test_prediction))
            #     plt.show()
            #     print("Saving image")
            #     if test_prediction == 1:
            #         filename = 'feature_test_img_car.png'
            #     else:
            #         filename = 'feature_test_img_nocar.png'

            #     mpimg.imsave(os.path.join(ROOT_DIR, 'unit_tests', 'test_images', filename), subimg)
            #     print("Saved image")
            #     plt.plot(list(range(len(test_features.T))), test_features.T)
            #     plt.show()

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                hot_windows.append(((xbox_left + xstart, ytop_draw + ystart),
                                    (xbox_left + xstart + win_draw, ytop_draw + win_draw + ystart)))

    return hot_windows

import numpy as np
import cv2
from skimage.feature import hog



def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features


def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32):  # bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


