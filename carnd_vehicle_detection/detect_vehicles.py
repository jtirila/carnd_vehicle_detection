# FIXME: This file will be the main entry point to the code. The detect_vehicles() function will be the
# FIXME: thing to run once everything is in place

import os

from moviepy.editor import VideoFileClip

from carnd_vehicle_detection import ROOT_DIR
# This is the default video name.
from carnd_vehicle_detection.classify.svm_classifier import get_classifier
from carnd_vehicle_detection.traverse_image.sliding_windows import slide_window
from carnd_vehicle_detection.traverse_image.search_windows import search_windows
from carnd_vehicle_detection.visualize import draw_boxes

PROJECT_VIDEO_PATH = os.path.join(ROOT_DIR, 'project_video.mp4')
_PROJECT_OUTPUT_PATH = os.path.join(ROOT_DIR, 'transformed.mp4')
_DEFAULT_CLASSIFIER_PATH = os.path.join(ROOT_DIR, 'classifier.p')


def detect_vehicles(input_video_path=PROJECT_VIDEO_PATH, output_path=_PROJECT_OUTPUT_PATH,
                    previous_classifier_path=_DEFAULT_CLASSIFIER_PATH, new_classifier_path=_DEFAULT_CLASSIFIER_PATH,
                    classifier_training_data=None):
    """Process the video whose path is provided as input.
    
    :param input_video_path: The path of a video file to be processed
    :param output_path: The path where to write the resulting video
    :param previous_classifier_path: The path of a pickled classifier, trained and saved previously. 
    :param new_classifier_path: The path for a new classifier, to be created. Only relevant if previous_classifier_path 
           is None
    :return: Nothing."""

    clip = VideoFileClip(input_video_path)
    container_for_detections = None  # FIXME
    if classifier_training_data is not None:
        training_data_params = [
            classifier_training_data['features_train'],
            classifier_training_data['labels_train'],
            classifier_training_data['features_valid'],
            classifier_training_data['labels_valid']
        ]
    else:
        training_data_params = (None, None, None, None)
    classifier_and_score = get_classifier(previous_classifier_path, new_classifier_path, *training_data_params)
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

    color_space = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 13  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16)  # Spatial binning dimensions
    hist_bins = 16  # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off
    y_start_stop = [360, None]  # Min and max in y to search in slide_window()

    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                           xy_window=(128, 128), xy_overlap=(0.7, 0.7))

    hot_windows = search_windows(image, windows, classifier, scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)

    window_image = draw_boxes(image, hot_windows, color=(0, 0, 255), thick=6)
    return window_image


if __name__ == "__main__":
    detect_vehicles(previous_classifier_path=_DEFAULT_CLASSIFIER_PATH)
