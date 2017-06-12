import os
import numpy as np
from copy import deepcopy
import cv2
from scipy.ndimage.measurements import label


# https://stackoverflow.com/q/13750447
from PIL import Image

from moviepy.editor import VideoFileClip

from carnd_vehicle_detection import ROOT_DIR
from carnd_vehicle_detection.classify import get_classifier
from carnd_vehicle_detection.traverse_image import slide_window, all_windows_divisible_by
from carnd_vehicle_detection.traverse_image import search_windows
from carnd_vehicle_detection.visualize import draw_boxes
from carnd_vehicle_detection.utils.find_cars import find_cars
from carnd_vehicle_detection.preprocess.color_convert import normalize_luminosity

# This is the default video to read as input.
# PROJECT_VIDEO_PATH = os.path.join(ROOT_DIR, 'project_video.mp4')
# PROJECT_VIDEO_PATH = os.path.join(ROOT_DIR, 'unit_tests', 'test_videos', 'subclip_0__15.mp4')
# PROJECT_VIDEO_PATH = os.path.join(ROOT_DIR, 'unit_tests', 'test_videos', 'subclip_15__20.mp4')
PROJECT_VIDEO_PATH = os.path.join(ROOT_DIR, 'unit_tests', 'test_videos', 'subclip_35__36.mp4')

_PROJECT_OUTPUT_PATH = os.path.join(ROOT_DIR, 'transformed.mp4')
_DEFAULT_CLASSIFIER_PATH = os.path.join(ROOT_DIR, 'classifier.p')


EXTRACT_PARAMS = {
    'color_space': 'YCrCb',
    'orient': 9,
    'pix_per_cell': 8,
    'cell_per_block': 2,
    'hog_channel': "ALL",
    'spatial_size': (32, 32),
    'hist_bins': 32,
    'spatial_feat': True,
    'hist_feat': True,
    'hog_feat': True
}

# Some windowing params
# NOTE: these must be very carefully selected! The start parameters in both directions must be exact multiples of
# NOTE: scale * EXTRACT_PARAMS['pix_per_cell'] for the lookup from global single-pass hog matrix to make any sense.
# NOTE: Otherwise there will be a shift in the values that makes any feature vector comparisons and hence also
# NOTE: classifications meaningless. For stop parameters, this is not so important though.
_Y_STARTS_STOPS_PER_SCALE = {1: [400, 592], 1.5: [396, 592], 2: [384, 672], 2.5: [400, 680]}
_X_STARTS_STOPS_PER_SCALE = {1: [536, 744], 1.5: [384, 900], 2: [160, 1112], 2.5: [None, None]}
_XY_WINDOW = (64, 64)
_XY_OVERLAP = (0.75, 0.75)
_SCALES = (1, 1.5, 2, 2.5)  # (1, 1.5, 2, 2.5)


def detect_vehicles(input_video_path=PROJECT_VIDEO_PATH, output_path=_PROJECT_OUTPUT_PATH,
                    previous_classifier_path=_DEFAULT_CLASSIFIER_PATH, new_classifier_path=_DEFAULT_CLASSIFIER_PATH,
                    classifier_training_data=None):
    """Process the video whose path is provided as input.
    
    :param input_video_path: The path of a video file to be processed
    :param output_path: The path where to write the resulting video
    :param previous_classifier_path: The path of a pickled classifier, trained and saved previously. 
    :param new_classifier_path: The path for a new classifier, to be created. Only relevant if previous_classifier_path 
           is None
    :param classifier_training_data: a dict containing previously prepared training and validation data       
    :return: Nothing."""

    clip = VideoFileClip(input_video_path)
    container_for_detections = None  # FIXME instantiate something for every detected vehicle and keep them in this
                                     # FIXME container, updating some values to track the vehicle (and also
                                     # FIXME discard apparent spurious detections
    if classifier_training_data is not None:
        assert isinstance(classifier_training_data, dict)
        training_data_params = [
            classifier_training_data['features_train'],
            classifier_training_data['labels_train'],
            classifier_training_data['features_valid'],
            classifier_training_data['labels_valid']
        ]
    else:
        training_data_params = (None, None, None, None)
    classifier_and_score = get_classifier(previous_classifier_path, new_classifier_path, *training_data_params,
                                          extract_features_dict=EXTRACT_PARAMS)
    classifier = classifier_and_score['classifier']
    scaler = classifier_and_score['scaler']
    transformed_clip = clip.fl_image(lambda image: _process_image(image, classifier, scaler, container_for_detections))
    transformed_clip.write_videofile(output_path, audio=False)


def _process_image(image, classifier, scaler, fixme_container_of_previous_detections_or_whatever):
    """Augments the image with superimposed vehicle detection results. This function orchestrates most of the heavy 
    lifting to detect lane lines. The steps performed to achieve this result are: 
       - FIXME Preprocess the images as follows: 
       - FIXME Perform the moving window search with multiple scales (... or whatever) to detect vehicles
       - FIXME Add the raw detections to ... (some data structures used for tracking the vehicles across frames)
       - FIXME Perform (... some analysis in the data structure classes to filter out outliers, find the 
               trajectories of vehicles... or whatever)
       - FIXME After the previous filtering step, draw the bounding boxes around the reliable detections and 
               superimpose them on the original image
         
    :param image: A rbg image. Use distortion corrected images. 
    :param fixme_container_of_previous_detections_or_whatever: Whatever. Something that contains info on the previous
           detections
    :return: an rgb image containing the detection visualizations"""

    # TODO: there is currently a redundant wrapper, get rid of it

    # Use the default params specified at the start of the file

    return _search_for_cars(image, classifier, scaler)


def _search_for_cars(raw_image, classifier, scaler, scales=_SCALES,
                     y_starts_stops=_Y_STARTS_STOPS_PER_SCALE, x_starts_stops=_X_STARTS_STOPS_PER_SCALE,
                     xy_window=_XY_WINDOW, xy_overlap=_XY_OVERLAP,
                     extract_params_orig=deepcopy(EXTRACT_PARAMS)):
    """Using a moving windows method at different scales, traverses the image and looks for vehicle detections.
    Returns an image with boxes drawn around detected vehicles.
    
    :param raw_image: The original image
    :param classifier: A classifier that has a predict() interface, supposed to be a LinearSVC
    :param scaler: The input scaler that was used to scale the training data
    :param scales: The scales to use for windows sizes, as multiples of the original xy window size 
    :param y_starts_stops: The start and stop pixel counts in y direction, a dict by scale like: 
           {1: [380, 580], 2: [400, None]}
    :param x_starts_stops: See previous line, just now for x direction
    :param xy_window: the base window size, as a (width, height) tuple
    :param xy_overlap: A Tuple with fractional values between 0 and 1 for how much neighboring windows are to 
           overlap
    :param extract_params_orig: A dict with all the parameters one wishes to set for extract_features, see its 
           documentation for what is available"""

    window_image = np.copy(raw_image)
    image = normalize_luminosity(window_image)

    hot_windows = []
    for scale in scales:
        y_start_stop, x_start_stop = y_starts_stops[scale], x_starts_stops[scale]

        xy_win = tuple([int(np.product(x)) for x in zip(xy_window, (scale,) * 2)])

        # return find_cars(image, y_start_stop[0], y_start_stop[1], 1, classifier, scaler, orient, pix_per_cell,
        #                  cell_per_block, spatial_size, hist_bins, color_space)

        hot_windows.extend(find_cars(image, y_start_stop[0], y_start_stop[1], scale, classifier, scaler, **EXTRACT_PARAMS))
        continue
        scale_pix_per_cell = int(scale * extract_params_orig['pix_per_cell'])

        windows = slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop,
                               xy_window=xy_win, xy_overlap=xy_overlap)

        # Assert that the pix per cell used for hog divides window size, step size and starting point in both
        # directions. This is essential for the hog feature alignments to work.
        assert all_windows_divisible_by(windows,  scale_pix_per_cell, x_start_stop[0], y_start_stop[0])

        # Remember these are copied over so changes not seen on module level. Not that it matters anyway.
        extract_params = deepcopy(extract_params_orig)
        extract_params['pix_per_cell'] = scale_pix_per_cell

        hot_windows.extend([on_window for on_window in search_windows(
            image, windows, classifier, scaler, **extract_params
        ) if on_window is not None])

        box_color = {1: (255, 0, 0), 1.5: (0, 255, 0), 2: (0, 0, 255),
                     2.5: (128, 0, 128), 3: (0, 0, 0), 4: (255, 255, 255)}[scale]
        window_image = draw_boxes(window_image, hot_windows, color=box_color, thick=3)

    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    heat = add_heat(heat, hot_windows)
    heatmap = np.clip(heat, 0, 255)
    # heamap = apply_threshold(heatmap, 1)
    # labels = label(heatmap)
    return heatmap
    # return draw_labeled_bboxes(np.copy(image), labels)

    for box in hot_windows:
        if box[1][0] - box[0][0] > 128:
            box_color = (255, 0, 0)
        elif  box[1][0] - box[0][0] > 96:
            box_color = (0, 255, 0)
        elif  box[1][0] - box[0][0] > 64:
            box_color = (0, 0, 255)
        else:
            box_color = (128, 0, 128)
        cv2.rectangle(raw_image, box[0],
                      box[1], box_color, 2)
    return raw_image


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


if __name__ == "__main__":
    detect_vehicles(previous_classifier_path=_DEFAULT_CLASSIFIER_PATH)
    # detect_vehicles(previous_classifier_path=None)
