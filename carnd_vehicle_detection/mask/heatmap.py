import numpy as np
from scipy.ndimage.measurements import label


def add_labeled_heatmap(image, hot_windows):
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    heat = _add_heat(heat, hot_windows)
    heatmap = np.clip(heat, 0, 255)
    heatmap = _apply_threshold(heatmap, 3)
    labels = label(heatmap)
    return labels


def _add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def _apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap