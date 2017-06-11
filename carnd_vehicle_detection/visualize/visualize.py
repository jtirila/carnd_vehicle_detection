import cv2
import numpy as np


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """Draw rectangles on an image
    :param img: An image, format does not matter
    :param bboxes: A collection of ((top_left_x, top_left_y), (bottom_right_x, bottom_right_y))
           bounding box specifiers
    :param color: a (r, g, b) tuple for line color
    :param thick: a positive integer for line thickness
    
    :return: a copy of the image with the bounding boxes drawn on top"""

    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy
