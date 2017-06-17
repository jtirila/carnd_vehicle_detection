import numpy as np
import cv2

SMOOTHING_KERNEL = np.ones((8, 8))
SMOOTHING_KERNEL[2:-2, 2:-2] = 2
SMOOTHING_KERNEL[3:-3, 3:-3] = 4
SMOOTHING_KERNEL = SMOOTHING_KERNEL / np.sum(SMOOTHING_KERNEL)

class AggregatedHeatmap:
    def __init__(self):
        self.smoothed_heatmaps = np.zeros((7, 720, 1280))

    def process_new_heatmap(self, heatmap):
        self.smoothed_heatmaps = np.roll(self.smoothed_heatmaps, 1, 0)
        self.smoothed_heatmaps[0] = self.smooth_heatmap(heatmap)


    @staticmethod
    def smooth_heatmap(heatmap):
        return cv2.filter2D(heatmap, -1, SMOOTHING_KERNEL)

    def smoothed_heatmap(self):
        return np.average(self.smoothed_heatmaps, 0, [50, 40, 30, 25, 25, 20, 20])  \
               * np.count_nonzero(self.smoothed_heatmaps, 0)
