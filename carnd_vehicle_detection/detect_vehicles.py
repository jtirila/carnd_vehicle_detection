# FIXME: This file will be the main entry point to the code. The detect_vehicles() function will be the
# FIXME: thing to run once everything is in place

import os

from moviepy.editor import VideoFileClip

from carnd_vehicle_detection import ROOT_DIR
# This is the default video name.
from carnd_vehicle_detection.classify.svm_classifier import get_classifier

_PROJECT_VIDEO_PATH = os.path.join(ROOT_DIR, 'project_video.mp4')
_PROJECT_OUTPUT_PATH = os.path.join(ROOT_DIR, 'transformed.mp4')
_DEFAULT_CLASSIFIER_PATH = os.path.join(ROOT_DIR, 'classifier.p')



def detect_vehicles(input_video_path=_PROJECT_VIDEO_PATH, output_path=_PROJECT_OUTPUT_PATH, classifier_path=None):
    """Process the video whose path is provided as input.
    
    :param input_video_path: The path of a video file to be processed
    :param output_path: The path where to write the resulting video
    :param classifier_path: The path of a pickled classifier, trained and saved previously. If none is provided, 
           the classifier will be trained from scratch and saved in the default path defined above as a constant
    :return: Nothing."""

    clip = VideoFileClip(input_video_path)
    container_for_detections = None # FIXME
    classifier_and_score = get_classifier(classifier_path)
    classifier = classifier_and_score['classifier']
    # transformed_clip = clip.fl_image(lambda image: _process_image(image, container_for_detections))
    # transformed_clip.write_videofile(output_path, audio=False)


def _process_image(image, fixme_container_of_previous_detections_or_whatever):
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
    return image


if __name__ == "__main__":
    detect_vehicles()
