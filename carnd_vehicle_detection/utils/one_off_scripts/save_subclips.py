from moviepy.editor import VideoFileClip
from carnd_vehicle_detection.detect_vehicles import PROJECT_VIDEO_PATH
from carnd_vehicle_detection import ROOT_DIR
import os

SUBCLIP_OUTPUT_DIRECTORY = os.path.join(ROOT_DIR, 'unit_tests', 'test_videos')


def save_subclips(video_path=PROJECT_VIDEO_PATH):
    clip = VideoFileClip(video_path)
    subclip = clip.subclip(35, 36)
    subclip.write_videofile(os.path.join(SUBCLIP_OUTPUT_DIRECTORY, 'subclip_35__36.mp4'),
                            audio=False)
    for start in (0, 15, 30):
        subclip = clip.subclip(start, start + 5)
        subclip.write_videofile(os.path.join(SUBCLIP_OUTPUT_DIRECTORY, 'subclip_{}__{}.mp4'.format(start, start + 5)),
                                audio=False)

    # Save also one longer piece
    subclip = clip.subclip(0, 15)
    subclip.write_videofile(os.path.join(SUBCLIP_OUTPUT_DIRECTORY, 'subclip_0__15.mp4'), audio=False)


if __name__ == "__main__":
    save_subclips()