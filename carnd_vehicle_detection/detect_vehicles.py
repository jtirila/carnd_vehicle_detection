import os
from copy import deepcopy
import numpy as np
from moviepy.editor import VideoFileClip

from carnd_vehicle_detection import ROOT_DIR
from carnd_vehicle_detection.classify import get_classifier
from carnd_vehicle_detection.mask import add_labeled_heatmap
from carnd_vehicle_detection.preprocess import normalize_luminosity
from carnd_vehicle_detection.models import AggregatedHeatmap
from carnd_vehicle_detection.traverse_image import find_cars
from carnd_vehicle_detection.visualize import draw_labeled_bboxes

DEFAULT_INPUT_VIDEO_PATH = os.path.join(ROOT_DIR, 'project_video.mp4')
# DEFAULT_INPUT_VIDEO_PATH = os.path.join(ROOT_DIR, 'unit_tests', 'test_videos', 'subclip_0__15.mp4')
# DEFAULT_INPUT_VIDEO_PATH = os.path.join(ROOT_DIR, 'unit_tests', 'test_videos', 'subclip_15__20.mp4')
# DEFAULT_INPUT_VIDEO_PATH = os.path.join(ROOT_DIR, 'unit_tests', 'test_videos', 'subclip_20__25.mp4')
# DEFAULT_INPUT_VIDEO_PATH = os.path.join(ROOT_DIR, 'unit_tests', 'test_videos', 'subclip_20__35.mp4')
# DEFAULT_INPUT_VIDEO_PATH = os.path.join(ROOT_DIR, 'unit_tests', 'test_videos', 'subclip_23__25.mp4')
# DEFAULT_INPUT_VIDEO_PATH = os.path.join(ROOT_DIR, 'unit_tests', 'test_videos', 'subclip_15__30.mp4')
# DEFAULT_INPUT_VIDEO_PATH = os.path.join(ROOT_DIR, 'unit_tests', 'test_videos', 'subclip_35__36.mp4')
# DEFAULT_INPUT_VIDEO_PATH = os.path.join(ROOT_DIR, 'unit_tests', 'test_videos', 'subclip_35__35_3.mp4')

_DEFAULT_VIDEO_OUTPUT_PATH = os.path.join(ROOT_DIR, 'transformed.mp4')
_DEFAULT_CLASSIFIER_PATH = os.path.join(ROOT_DIR, 'classifier.p')


DEFAULT_EXTRACT_PARAMS = {
    'color_space': 'YCrCb',
    'orient': 9,
    'pix_per_cell': 8,
    'cell_per_block': 2,
    'hog_channel': "ALL",
    'spatial_size': (32, 32),
    'hist_bins': 16,
    'spatial_feat': True,
    'hist_feat': True,
    'hog_feat': True
}

_DEFAULT_Y_STARTS_STOPS_PER_SCALE = {
    0.5: [400, 600],
    1: [400, 600],
    1.4: [400, 670],
    1.5: [400, 670],
    1.7: [400, 670],
    2: [430, 670],
    2.2: [430, 670],
    2.5: [430, 670],
    3: [430, 670]}

_DEFAULT_X_STARTS_STOPS_PER_SCALE = {
    0.5: [300, 980],
    1: [150, 1180],
    1.4: [150, None],
    1.5: [150, None],
    1.7: [150, None],
    2: [150, None],
    2.2: [150, None],
    2.5: [150, None],
    3: [150, None]}

_DEFAULT_SCALES = (1, 1.5, 1.7, 2)


def detect_vehicles(input_video_path=DEFAULT_INPUT_VIDEO_PATH, output_path=_DEFAULT_VIDEO_OUTPUT_PATH,
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
                                          extract_features_dict=DEFAULT_EXTRACT_PARAMS)
    classifier = classifier_and_score['classifier']
    scaler = classifier_and_score['scaler']
    ahm = AggregatedHeatmap()
    transformed_clip = clip.fl_image(lambda image: search_for_cars(image, classifier, scaler, ahm))
    transformed_clip.write_videofile(output_path, audio=False)


def search_for_cars(raw_image, classifier, scaler, aggregated_heatmp, scales=_DEFAULT_SCALES,
                    y_starts_stops=_DEFAULT_Y_STARTS_STOPS_PER_SCALE, x_starts_stops=_DEFAULT_X_STARTS_STOPS_PER_SCALE,
                    extract_params=deepcopy(DEFAULT_EXTRACT_PARAMS)):
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

    draw_img = np.copy(raw_image)
    # plt.imshow(raw_image)
    # plt.show()
    image = normalize_luminosity(raw_image)
    # plt.imshow(image)
    # plt.show()


    hot_windows = []
    for scale in scales:
        y_start_stop, x_start_stop = y_starts_stops[scale], x_starts_stops[scale]
        hot_windows.extend(
            # find_cars(image, y_start_stop, y_start_stop, scale, classifier, scaler, extract_params)
            find_cars(image, *y_start_stop, *x_start_stop, scale, classifier, scaler, **extract_params)
        )
    high_confidence_labels, low_confidence_labels = add_labeled_heatmap(image, hot_windows, aggregated_heatmp)
    return draw_labeled_bboxes(draw_img, (high_confidence_labels, ))  # low_confidence_labels


if __name__ == "__main__":
    # detect_vehicles(previous_classifier_path=None, output_path="trans.mp4")
    # detect_vehicles(previous_classifier_path=_DEFAULT_CLASSIFIER_PATH, output_path="trans.mp4")
    # detect_vehicles(previous_classifier_path=_DEFAULT_CLASSIFIER_PATH)
    detect_vehicles(previous_classifier_path=None)

