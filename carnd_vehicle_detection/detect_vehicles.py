import os
from copy import deepcopy

import numpy as np
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label

from carnd_vehicle_detection import ROOT_DIR
from carnd_vehicle_detection.classify import get_classifier
from carnd_vehicle_detection.mask import add_labeled_heatmap
from carnd_vehicle_detection.preprocess import normalize_luminosity
# This is the default video to read as input.
from carnd_vehicle_detection.traverse_image import find_cars
from carnd_vehicle_detection.visualize import draw_labeled_bboxes

# PROJECT_VIDEO_PATH = os.path.join(ROOT_DIR, 'project_video.mp4')
# PROJECT_VIDEO_PATH = os.path.join(ROOT_DIR, 'unit_tests', 'test_videos', 'subclip_0__15.mp4')
PROJECT_VIDEO_PATH = os.path.join(ROOT_DIR, 'unit_tests', 'test_videos', 'subclip_15__20.mp4')
# PROJECT_VIDEO_PATH = os.path.join(ROOT_DIR, 'unit_tests', 'test_videos', 'subclip_15__30.mp4')
# PROJECT_VIDEO_PATH = os.path.join(ROOT_DIR, 'unit_tests', 'test_videos', 'subclip_35__36.mp4')
# PROJECT_VIDEO_PATH = os.path.join(ROOT_DIR, 'unit_tests', 'test_videos', 'subclip_35__35_3.mp4')

_PROJECT_OUTPUT_PATH = os.path.join(ROOT_DIR, 'transformed.mp4')
_DEFAULT_CLASSIFIER_PATH = os.path.join(ROOT_DIR, 'classifier.p')


EXTRACT_PARAMS = {
    'color_space': 'YCrCb',
    'orient': 9,
    'pix_per_cell': 8,
    'cell_per_block': 2,
    'hog_channel': 0,
    'spatial_size': (16, 16),
    'hist_bins': 32,
    'spatial_feat': False,
    'hist_feat': True,
    'hog_feat': True
}

_DEFAULT_Y_STARTS_STOPS_PER_SCALE = {
    0.5: [410, 450],
    1: [410, 530],
    1.5: [410, 690],
    2: [410, 690],
    2.2: [410, 690],
    2.5: [410, 690],
    3: [410, 690]}
_DEFAULT_X_STARTS_STOPS_PER_SCALE = {
    0.5: [600, 1080],
    1: [700, 1140],
    1.5: [600, 1100],
    2: [400, None],
    2.2: [400, None],
    2.5: [300, None],
    3: [300, None]}
_DEFAULT_SCALES = (1, 1.5, 2, 2.2, 2.5, 3)


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
    transformed_clip = clip.fl_image(lambda image: search_for_cars(image, classifier, scaler))
    transformed_clip.write_videofile(output_path, audio=False)


def search_for_cars(raw_image, classifier, scaler, scales=_DEFAULT_SCALES,
                    y_starts_stops=_DEFAULT_Y_STARTS_STOPS_PER_SCALE, x_starts_stops=_DEFAULT_X_STARTS_STOPS_PER_SCALE,
                    extract_params=deepcopy(EXTRACT_PARAMS)):
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
    :param extract_params: A dict with all the parameters one wishes to set for extract_features, see its 
           documentation for what is available"""

    image = normalize_luminosity(raw_image)

    hot_windows = []
    for scale in scales:
        y_start_stop, x_start_stop = y_starts_stops[scale], x_starts_stops[scale]
        hot_windows.extend(
            find_cars(image, y_start_stop, x_start_stop, scale, classifier, scaler, extract_params)
        )
    labels = add_labeled_heatmap(image, hot_windows)
    return draw_labeled_bboxes(raw_image, labels)


if __name__ == "__main__":
    # detect_vehicles(previous_classifier_path=_DEFAULT_CLASSIFIER_PATH)
    detect_vehicles(previous_classifier_path=None)

