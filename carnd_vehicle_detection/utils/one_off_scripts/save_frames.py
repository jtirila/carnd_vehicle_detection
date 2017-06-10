from carnd_vehicle_detection import PROJECT_VIDEO_PATH
from moviepy.editor import VideoFileClip
from unit_tests import TEST_IMG_DIR
from glob import glob
import matplotlib.image as mpimg

import cv2
import os


def save_some_frames_and_subframes():
    cap = cv2.VideoCapture(PROJECT_VIDEO_PATH)

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
            saved_pair_count += 1
            mpimg.imsave(os.path.join(TEST_IMG_DIR, full_filename), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mpimg.imsave(os.path.join(TEST_IMG_DIR, sub_filename_1), cv2.cvtColor(frame[:65, :65, :], cv2.COLOR_BGR2RGB))
            mpimg.imsave(os.path.join(TEST_IMG_DIR, sub_filename_2), cv2.cvtColor(frame[410:475, 130:195, :], cv2.COLOR_BGR2RGB))
        else:
            continue
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    save_some_frames_and_subframes()

