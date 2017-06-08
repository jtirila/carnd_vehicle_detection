import numpy as np
import matplotlib.image as mpimg
import cv2
from sklearn.preprocessing import StandardScaler
from carnd_vehicle_detection.preprocess import get_hog_features, color_hist, bin_spatial, color_convert


def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, hist_range=(0, 256), orient=9, pix_per_cell=8, cell_per_block=2, spatial_feat=True, hist_feat=True, hog_feat=True, hog_channel="ALL", vis=False, feature_vec=True):
    # Create a list to append feature vectors to
    features = []
    for img_orig in imgs:
        img_features = []
        # img_orig = mpimg.imread(img_filename)
        assert color_space in ("RGB", "HLS", "HSV", "LUV", "GRAY")
        if not color_space == "RGB":
            colorspace_val = getattr(cv2, "COLOR_RGB2{}".format(color_space))
            img = cv2.cvtColor(img_orig, colorspace_val)
        else:
            img = img_orig

        if spatial_feat:
            img_features.append(bin_spatial(img, color_space, spatial_size))

        if hist_feat:
            img_features.append(color_hist(img, hist_bins, hist_range))

        if hog_feat:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(img.shape[2]):
                    hog_features.extend(get_hog_features(img[:, :, channel], orient, pix_per_cell, cell_per_block,
                                                         vis=vis, feature_vec=feature_vec))
            else:
                hog_features = get_hog_features(img[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=vis, feature_vec=feature_vec)
            img_features.append(hog_features)

        img_features_flat = np.concatenate(img_features)

        features.append(img_features_flat)

        # Iterate through the list of images
        # Read in each one by one
        # apply color conversion if other than 'RGB'
        # Apply bin_spatial() to get spatial color features
        # Apply color_hist() to get color histogram features
        # Append the new feature vector to the features list
        # Return list of feature vectors
    return features

# def extract_features(imgs):
#     """Extracts the features from the images collection
#
#     :param imgs: A numpy ndarray containing the rgb images
#     :return: A numpy ndarray with the transformed features
#     """
#
#     # FIXME
#     transformed = np.array([img.ravel() for img in imgs])
#     return transformed


# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        global_hog_features=None,
                        global_color_converted=None,
                        spatial_feat=True, hist_feat=True, hog_feat=True):

    """FIXME_ document. At this point, this is just plain copy-paste from lab.
    
    :param global_hog_features"""

    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
    feature_image = color_convert(img, color_space)
    # 3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            local_hog_features =
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)


def single_img_features_generator(img, windows, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):


    """FIXME_ document. At this point, this is just plain copy-paste from lab.
    
    :param global_hog_features"""
    global_hog_features = get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
    global_color_converted = color_convert(img, color_space)

    for windows in windows:
        # 1) Define an empty list to receive features
        img_features = []
        # 2) Apply color conversion if other than 'RGB'
        feature_image = color_convert(img, color_space)
        # 3) Compute spatial features if flag is set
        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            # 4) Append features to list
            img_features.append(spatial_features)
        # 5) Compute histogram features if flag is set
        if hist_feat == True:
            hist_features = color_hist(feature_image, nbins=hist_bins)
            # 6) Append features to list
            img_features.append(hist_features)
        # 7) Compute HOG features if flag is set
        if hog_feat == True:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # 8) Append features to list
            img_features.append(hog_features)

        # 9) Return concatenated array of features
        yield np.concatenate(img_features)

def scale_features(features):
    """Apply StandardScaler to features to get a normalized set of feature vectors.
    
    :param features: a numpy ndarray, with each element representing one observation
    :return: a numpy ndarray of same dimensions, with the values normalized as per 
             the Scikit-learn StandardScaler."""
    scaler = StandardScaler().fit(features)
    # Apply the scaler
    scaled_features = scaler.transform(features)
    return scaled_features


