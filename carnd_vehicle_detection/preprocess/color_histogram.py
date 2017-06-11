import numpy as np

def color_hist(image, nbins=32, bins_range=(0, 256)):
    """Compute the color histogram of a 3-channel image
    
    :param image: the original image
    :param nbins: The number of histogram bins to divide the color range into
    :param bins_range: The range of values
    
    :returns: The histogram values concatenated into a feature vector"""
    # Compute the histogram of the RGB channels separately
    try:
        rhist = np.histogram(image[:, :, 0], bins=nbins, range=bins_range)
        ghist = np.histogram(image[:, :, 1], bins=nbins, range=bins_range)
        bhist = np.histogram(image[:, :, 2], bins=nbins, range=bins_range)
    # Generating bin centers
    except IndexError:
        pass

    bin_edges = rhist[1]
    bin_centers = ((np.roll(bin_edges, 1) + bin_edges) / 2)[1:]

    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    return hist_features
