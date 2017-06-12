import cv2
import numpy as np
from matplotlib import pyplot as plt


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


def one_by_two_plot(first, second, first_cmap=None, second_cmap=None, first_title="First image", second_title="Second image"):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.set_title(first_title)
    if first_cmap is not None:
        ax1.imshow(first, cmap=first_cmap)
    else:
        ax1.imshow(first)

    ax2.set_title(second_title)
    if second_cmap is not None:
        ax2.imshow(second, cmap=second_cmap)
    else:
        ax2.imshow(second)
    plt.show()
