import unittest
import os
from unit_tests import TEST_IMG_DIR
from carnd_vehicle_detection.preprocess import read_training_data, extract_features
from carnd_vehicle_detection import detect_vehicles

VEHICLES_TEST_IMAGE_EXPR = os.path.join(TEST_IMG_DIR, 'vehicles', '**', '*png')
NONVEHICLES_TEST_IMAGE_EXPR = os.path.join(TEST_IMG_DIR, 'nonvehicles', '**', '*.png')

_TEST_INPUT_VIDEO_PATH=os.path.join(os.path.dirname(__file__), 'test_videos', 'subclip_30__35.mp4')
_TEST_OUTPUT_VIDEO_PATH=os.path.join(os.path.dirname(__file__), 'test_videos', 'subclip_30__35_processed.mp4')

_TEST_EXCERPT_INPUT_VIDEO_PATH = os.path.join(os.path.dirname(__file__), 'test_videos', 'subclip_35__36.mp4')
_TEST_EXCERPT_OUTPUT_VIDEO_PATH = os.path.join(os.path.dirname(__file__), 'test_videos', 'subclip_35__36_processed.mp4')

_TEST_LONG_INPUT_VIDEO_PATH=os.path.join(os.path.dirname(__file__), 'test_videos', 'subclip_0__15.mp4')
_TEST_LONG_OUTPUT_VIDEO_PATH=os.path.join(os.path.dirname(__file__), 'test_videos', 'subclip_0__15_processed.mp4')


def _remove_output_videos():
    for path in (_TEST_OUTPUT_VIDEO_PATH, _TEST_EXCERPT_OUTPUT_VIDEO_PATH, _TEST_LONG_OUTPUT_VIDEO_PATH):
        try:
            # Try to remove the output video, just in case it has been left hanging around
            os.remove(path)
        except FileNotFoundError:
            # No need to do anything
            pass

class TestDetectVehiclesPipelineDoesNotFail(unittest.TestCase):

    def setUp(self):
        _remove_output_videos()

    def test_detect_vehicles_excerpt_fails_not(self):
        features_train, features_valid, labels_train, labels_valid = read_training_data(VEHICLES_TEST_IMAGE_EXPR,
                                                                                        NONVEHICLES_TEST_IMAGE_EXPR)
        self.assertFalse(os.path.exists(_TEST_EXCERPT_OUTPUT_VIDEO_PATH))
        detect_vehicles(_TEST_EXCERPT_INPUT_VIDEO_PATH, _TEST_EXCERPT_OUTPUT_VIDEO_PATH, None, None,
                        {'features_train': features_train, 'labels_train': labels_train,
                         'features_valid': features_valid, 'labels_valid': labels_valid})
        self.assertTrue(os.path.exists(_TEST_EXCERPT_OUTPUT_VIDEO_PATH))

    def test_detect_vehicles_fails_not(self):
        features_train, features_valid, labels_train, labels_valid = read_training_data(VEHICLES_TEST_IMAGE_EXPR,
                                                                                        NONVEHICLES_TEST_IMAGE_EXPR)
        self.assertFalse(os.path.exists(_TEST_OUTPUT_VIDEO_PATH))
        detect_vehicles(_TEST_INPUT_VIDEO_PATH, _TEST_OUTPUT_VIDEO_PATH, None, None,
                        {'features_train': features_train, 'labels_train': labels_train,
                         'features_valid': features_valid, 'labels_valid': labels_valid})
        self.assertTrue(os.path.exists(_TEST_OUTPUT_VIDEO_PATH))

    def test_detect_vehicles_long_clip_with_teaching_fails_not(self):
        # Read all of the training data (the
        features_train, features_valid, labels_train, labels_valid = read_training_data()
        self.assertFalse(os.path.exists(_TEST_LONG_OUTPUT_VIDEO_PATH))

        # Run the pipeline, not saving the classifier or scaler to file.
        detect_vehicles(_TEST_LONG_INPUT_VIDEO_PATH, _TEST_LONG_OUTPUT_VIDEO_PATH, None, None)
        self.assertTrue(os.path.exists(_TEST_LONG_OUTPUT_VIDEO_PATH))
