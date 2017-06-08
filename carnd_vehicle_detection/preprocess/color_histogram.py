import numpy as np

def color_hist(image, nbins=32, bins_range=(0, 256)):
    """Compute the color histogram of a 3-channel image
    
    :param image: the original image
    :param nbins: The number of histogram bins to divide the color range into
    :param bins_range: The range of values
    
    :returns: The histogram values concatenated into a feature vector"""

    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(image[:,:,0], bins=nbins)
    channel2_hist = np.histogram(image[:,:,1], bins=nbins)
    channel3_hist = np.histogram(image[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

