import numpy as np
from scipy.ndimage.measurements import label


def add_labeled_heatmap(image, hot_windows, aggregated_heatmap):
    """Processes the hot windows, counts heatmap values, applies the aggregate and returns (essentially) the 
    bounding boxes of the labels."""
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    heat = _add_heat(heat, hot_windows)
    heatmap = np.clip(heat, 0, 255)
    aggregated_heatmap.process_new_heatmap(heatmap)
    high_confidence_heatmap = _apply_threshold(aggregated_heatmap.smoothed_heatmap(), 3)
    low_confidence_heatmap = _apply_threshold(aggregated_heatmap.smoothed_heatmap(), 1)
    high_confidence_labels = label(high_confidence_heatmap)
    low_confidence_labels = label(low_confidence_heatmap)
    return high_confidence_labels, low_confidence_labels


def _add_heat(heatmap, bbox_list):
    """Computes the heatmap from the list of initial bounding boxes (windows with matches)."""
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def _apply_threshold(heatmap, threshold):
    """Returns a version of the heatmap where all the observations are zeroed out whose heatmap value is below 
    the threshold """
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap