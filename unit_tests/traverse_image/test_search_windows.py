from carnd_vehicle_detection.traverse_image import poppeli
from carnd_vehicle_detection.classify import get_classifier
from unit_tests import TEST_IMG_DIR
from matplotlib import image as mpimg
import os
import unittest

_TEST_IMAGE_PATH = os.path.join(TEST_IMG_DIR, 'test_frame_3.png')
_WIN_WIDTH = 96
_DELTA = _WIN_WIDTH / 4
_YMIN = 480
_YMAX = 576
_TEST_WINDOWS = [((n * _DELTA, _YMIN), ((n * _DELTA) + _WIN_WIDTH, _YMAX)) for n in range(40)]


class TestSearchWindows(unittest.TestCase):
    def test_detections_made(self):
        clf_data = get_classifier()
        img = mpimg.imread(_TEST_IMAGE_PATH)
        windows = poppeli(img, _TEST_WINDOWS, clf_data['classifier'], clf_data['scaler'], 'YCrCb',
                          pix_per_cell=12)
        assert any(windows)

