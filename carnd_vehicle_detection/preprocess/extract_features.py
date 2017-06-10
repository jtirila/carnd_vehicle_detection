import numpy as np
import matplotlib.image as mpimg
import cv2
from sklearn.preprocessing import StandardScaler
from carnd_vehicle_detection.preprocess import get_hog_features, color_hist, bin_spatial, convert_color


def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, hist_range=(0, 256), orient=9, pix_per_cell=8, cell_per_block=2, spatial_feat=True, hist_feat=True, hog_feat=True, hog_channel="ALL", vis=False, feature_vec=True):
    # Create a list to append feature vectors to
    features = []
    for img in imgs:
        img_features = single_img_features(img, color_space=color_space, spatial_size=spatial_size,
                                           hist_bins=hist_bins, img_hog_features=None,
                                           spatial_feat=spatial_feat, hist_feat=hist_feat,
                                           hog_feat=hog_feat, orient=orient, pix_per_cell=pix_per_cell,
                                           cell_per_block=cell_per_block, hog_channel=hog_channel)
        features.append(img_features)
    return features


# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32), hist_bins=32, img_hog_features=None,
                        spatial_feat=True, hist_feat=True, hog_feat=True, orient=9, pix_per_cell=8,
                        cell_per_block=2, hog_channel="ALL"):

    """FIXME_ document. At this point, this is just plain copy-paste from lab.
    
    :param global_hog_features"""

    # 1) Define an empty list to receive features
    img_copy = convert_color(img, color_space)
    img_features = []
    # 3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(img_copy, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(img_copy, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat == True:
        if img_hog_features is not None:
            img_features.append(img_hog_features.ravel())
        else:
            # FIXME: First attempting to use grayscaling. Other color conversions are possible too.
            if color_space != "GRAY":
                # Otherwise, the image is in the right format to begin with
                gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = np.copy(img_copy)
            hog_features = get_hog_features(gray_image, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)


