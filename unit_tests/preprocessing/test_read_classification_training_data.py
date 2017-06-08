from carnd_vehicle_detection.preprocessing.read_classification_training_data import read_training_data
from unit_tests import TEST_IMG_DIR
import os
import unittest

VEHICLES_TEST_IMAGE_EXPR = os.path.join(TEST_IMG_DIR, 'vehicles', '**', '*png')
NONVEHICLES_TEST_IMAGE_EXPR = os.path.join(TEST_IMG_DIR, 'nonvehicles', '**', '*.png')


class TestSomething(unittest.TestCase):

    def test_reading_test_data(self):
        data = read_training_data(VEHICLES_TEST_IMAGE_EXPR, NONVEHICLES_TEST_IMAGE_EXPR)
        self.assertEqual(len(data[0]) + len(data[1]), 20)
