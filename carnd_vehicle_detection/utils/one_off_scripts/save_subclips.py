from moviepy.editor import VideoFileClip
from carnd_vehicle_detection.detect_vehicles import DEFAULT_INPUT_VIDEO_PATH
from carnd_vehicle_detection import ROOT_DIR
import os

SUBCLIP_OUTPUT_DIRECTORY = os.path.join(ROOT_DIR, 'unit_tests', 'test_videos')


def save_subclip(video_path=DEFAULT_INPUT_VIDEO_PATH, start=0, end=10):
    clip = VideoFileClip(video_path)
    subclip = clip.subclip(start, end)
    subclip.write_videofile(os.path.join(SUBCLIP_OUTPUT_DIRECTORY, 'subclip_{}__{}.mp4'.format(start, end)),
                            audio=False)

if __name__ == "__main__":
    save_subclip(start=15, end=30)