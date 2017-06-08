from carnd_vehicle_detection.image_traversal.sliding_windows import slide_window
import unittest
import numpy as np


class TestSlidingWindowsDifferentOverlaps(unittest.TestCase):

    def test_with_no_overlap(self):
        img = np.zeros((64, 64, 3))
        win_list = slide_window(img, [None, None], [None, None], (32, 32), (0.0, 0.0))
        expected_win_list = [((0, 0), (32, 32)), ((32, 0), (64, 32)), ((0, 32), (32, 64)), ((32, 32), (64, 64))]
        self.assertEquals(win_list, expected_win_list)

    def test_with_much_overlap(self):
        img = np.zeros((64, 64, 3))
        win_list = slide_window(img, [None, None], [None, None], (32, 32), (0.75, 0.75))
        expected_win_list = [
            ((0, 0), (32, 32)), ((8, 0), (40, 32)), ((16, 0), (48, 32)), ((24, 0), (56, 32)), ((32, 0), (64, 32)),
            ((0, 8), (32, 40)), ((8, 8), (40, 40)), ((16, 8), (48, 40)), ((24, 8), (56, 40)), ((32, 8), (64, 40)),
            ((0, 16), (32, 48)), ((8, 16), (40, 48)), ((16, 16), (48, 48)), ((24, 16), (56, 48)), ((32, 16), (64, 48)),
            ((0, 24), (32, 56)), ((8, 24), (40, 56)), ((16, 24), (48, 56)), ((24, 24), (56, 56)), ((32, 24), (64, 56)),
            ((0, 32), (32, 64)), ((8, 32), (40, 64)), ((16, 32), (48, 64)), ((24, 32), (56, 64)), ((32, 32), (64, 64))
        ]
        self.assertEquals(win_list, expected_win_list)


class TestSlidingWindowsDifferentRemainders(unittest.TestCase):

    def test_with_remainder_x(self):
        img = np.zeros((64, 95, 3))
        win_list = slide_window(img, [None, None], [None, None], (32, 32), (0.0, 0.0))
        expected_win_list = [((0, 0), (32, 32)), ((32, 0), (64, 32)), ((0, 32), (32, 64)), ((32, 32), (64, 64))]
        self.assertEquals(win_list, expected_win_list)


    def test_with_remainder_y(self):
        img = np.zeros((70, 64, 3))
        win_list = slide_window(img, [None, None], [None, None], (32, 32), (0.0, 0.0))
        expected_win_list = [((0, 0), (32, 32)), ((32, 0), (64, 32)), ((0, 32), (32, 64)), ((32, 32), (64, 64))]
        self.assertEquals(win_list, expected_win_list)
