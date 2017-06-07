# FIXME: This file will be the main entry point to the code. The detect_vehicles() function will be the
# FIXME: thing to run once everything is in place

from carnd_vehicle_detection import ROOT_DIR
import os

# This is the default video name.
PROJECT_VIDEO_PATH = os.path.join(ROOT_DIR, 'project_video.mp4')


def detect_vehicles(input_video_path=PROJECT_VIDEO_PATH):
    """Process the video whose path is provided as input.
    
    :param input_video_path: The path of a video file to be processed.
    :return: Nothing."""


if __name__  == "__main__":
    detect_vehicles()
