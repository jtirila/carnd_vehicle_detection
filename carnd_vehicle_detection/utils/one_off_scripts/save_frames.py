from carnd_vehicle_detection import DEFAULT_INPUT_VIDEO_PATH
from moviepy.editor import VideoFileClip
from unit_tests import TEST_IMG_DIR
from glob import glob
import matplotlib.image as mpimg
import numpy as np

import cv2
import os


def save_some_frames_and_subframes():
    cap = cv2.VideoCapture(DEFAULT_INPUT_VIDEO_PATH)

    frame_count = 1
    saved_pair_count = 1
    while cap.isOpened():

        print("Frame number {}".format(frame_count))

        frame_count += 1

        if frame_count > 700:
            break

        ret, frame = cap.read()
        if frame_count in (376, 589, 680):
            full_filename = "test_frame_{}.png".format(saved_pair_count)
            sub_filename_1 = "test_subframe_{}_1.png".format(saved_pair_count)
            sub_filename_2 = "test_subframe_{}_2.png".format(saved_pair_count)
            sub_filename_3 = "test_subframe_{}_3.png".format(saved_pair_count)
            sub_filename_4 = "test_subframe_{}_4.png".format(saved_pair_count)
            sub_filename_5 = "test_subframe_{}_5.png".format(saved_pair_count)
            sub_filename_6 = "test_subframe_{}_6.png".format(saved_pair_count)
            saved_pair_count += 1
            mpimg.imsave(os.path.join(TEST_IMG_DIR, full_filename),
                         cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mpimg.imsave(os.path.join(TEST_IMG_DIR, sub_filename_1),
                         cv2.cvtColor(frame[:64, :64, :], cv2.COLOR_BGR2RGB))
            mpimg.imsave(os.path.join(TEST_IMG_DIR, sub_filename_2),
                         cv2.cvtColor(frame[410:474, 130:194, :], cv2.COLOR_BGR2RGB))
            mpimg.imsave(os.path.join(TEST_IMG_DIR, sub_filename_3),
                         cv2.cvtColor(frame[400:464, 128:192, :], cv2.COLOR_BGR2RGB))
            mpimg.imsave(os.path.join(TEST_IMG_DIR, sub_filename_4),
                         cv2.cvtColor(frame[400:464, 248:312, :], cv2.COLOR_BGR2RGB))
            mpimg.imsave(os.path.join(TEST_IMG_DIR, sub_filename_5),
                         cv2.cvtColor(frame[396:492, 120:216, :], cv2.COLOR_BGR2RGB))
            mpimg.imsave(os.path.join(TEST_IMG_DIR, sub_filename_6),
                         cv2.cvtColor(frame[420:516, 144:240, :], cv2.COLOR_BGR2RGB))
        else:
            continue
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    save_some_frames_and_subframes()

