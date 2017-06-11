import unittest
from carnd_vehicle_detection.models import DetectedVehicle


class TestDetectedVehicleWindowOverlappingDetector(unittest.TestCase):

    def test_completely_overlapping_windows(self):
        win1 = ((1, 1), (2, 2))
        win2 = ((1, 1), (2, 2))
        self.assertTrue(DetectedVehicle.windows_overlap(win1, win2))
        self.assertTrue(DetectedVehicle.windows_overlap(win2, win1))

    def test_non_overlapping_windows(self):
        win1 = ((1, 1), (2, 2))
        win2 = ((3, 3), (4, 4))
        self.assertFalse(DetectedVehicle.windows_overlap(win1, win2))
        self.assertFalse(DetectedVehicle.windows_overlap(win2, win1))

    def test_a_little_overlapping_windows(self):
        win1 = ((1, 1), (2, 2))
        win2 = ((1.9, 1.9), (3, 3))
        self.assertTrue(DetectedVehicle.windows_overlap(win1, win2))
        self.assertTrue(DetectedVehicle.windows_overlap(win2, win1))

    def test_windows_touching_at_corner(self):
        win1 = ((1, 1), (2, 2))
        win2 = ((2, 2), (3, 3))
        self.assertFalse(DetectedVehicle.windows_overlap(win1, win2))
        self.assertFalse(DetectedVehicle.windows_overlap(win2, win1))

