import numpy as np
from carnd_vehicle_detection.preprocess import get_hog_features, color_hist, bin_spatial, convert_color


def single_img_features(img, color_space='RGB', spatial_size=(32, 32), hist_bins=32, bins_range=(0, 256),
                        img_hog_features=None, spatial_feat=True, hist_feat=True, hog_feat=True, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel="ALL"):


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
        hist_features = color_hist(img_copy, nbins=hist_bins, bins_range=bins_range)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat == True:
        if img_hog_features is not None:
            img_features.append(img_hog_features.ravel())
        else:
            # FIXME: First attempting to use grayscaling. Other color conversions are possible too.
            hog_features = get_hog_features(img_copy[:, :, 2], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)


