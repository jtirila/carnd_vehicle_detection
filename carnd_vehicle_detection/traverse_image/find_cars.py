import numpy as np
from skimage.feature import hog
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import cv2
from carnd_vehicle_detection import ROOT_DIR
import os
from carnd_vehicle_detection.preprocess import normalize_luminosity, convert_color, bin_spatial, color_hist, get_hog_features

COUNTER = 0

def find_cars(img, ystart, ystop, xstart, xstop, scale, svc, X_scaler, color_space="RGB", hog_channel="ALL",
              orient=9, pix_per_cell=8, cell_per_block=2, spatial_size=(32, 32), hist_bins=32,
              hog_feat=True, hist_feat=True, spatial_feat=True):
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
    orig_resized = np.copy(img_tosearch)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))
        orig_resized = cv2.resize(orig_resized, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))


    t, f = True, False
    if f:
        plt.imshow(orig_resized)
        plt.title("Original image at scale {}".format(scale))
        plt.show()

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    # Jump around more for smaller scales, we don't need too reliable detections far away.
    if scale < 1.6:
        cells_per_step = 2  # Instead of overlap, define how many cells to step
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
            img_features = []
            # Extract HOG for this patch
            if hog_feat:
                if hog_channel == "ALL":
                    hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    img_features.append(np.hstack((hog_feat1, hog_feat2, hog_feat3)))
                else:
                    assert hog_channel in (0, 1, 2)
                    if hog_channel == 0:
                        img_features.append(hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel())
                    elif hog_channel == 1:
                        img_features.append(hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel())
                    else:
                        img_features.append(hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel())

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))
            subimg_orig = cv2.resize(orig_resized[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            if spatial_feat:
                img_features.append(bin_spatial(subimg, size=spatial_size))
            if hist_feat:
                img_features.append(color_hist(subimg, nbins=hist_bins))

            # Scale features and make a prediction
            raw_features = np.hstack(img_features).reshape(1, -1)
            test_features = X_scaler.transform(raw_features)
            test_prediction = svc.predict(test_features)
            if COUNTER == 1:
                pass
                # plt.imshow(subimg_orig)
                # plt.title("Original image, pred result: {}, scale: {}".format(test_prediction, scale))
                # plt.show()
                # plt.imshow(subimg)
                # plt.title("Original image color converted, pred result: {}".format(test_prediction))
                # plt.show()
                # plt.imshow(subimg[:, :, 0], cmap='gray')
                # plt.title("Predicdtion result: {}, channel: 0".format(test_prediction))
                # plt.show()
                # plt.imshow(subimg[:, :, 1], cmap='gray')
                # plt.title("Predicdtion result: {}, channel: 1".format(test_prediction))
                # plt.show()
                # plt.imshow(subimg[:, :, 2], cmap='gray')
                # plt.title("Predicdtion result: {}, channel: 2".format(test_prediction))
                # plt.show()
                # print("Saving image")
                # if test_prediction == 1:
                #     filename = 'feature_test_img_car.png'
                # else:
                #     filename = 'feature_test_img_nocar.png'

                # mpimg.imsave(os.path.join(ROOT_DIR, 'unit_tests', 'test_images', filename), subimg)
                # print("Saved image")
                # plt.plot(list(range(len(test_features.T))), test_features.T)
                # plt.show()

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                hot_windows.append(((xbox_left + xstart, ytop_draw + ystart),
                                    (xbox_left + xstart + win_draw, ytop_draw + win_draw + ystart)))

    return hot_windows

