import numpy as np
from carnd_vehicle_detection.preprocess import get_hog_features, color_hist, bin_spatial, convert_color


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
        if img_hog_features is not None:
            img_features.append(np.ravel(img_hog_features))
        else:
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


