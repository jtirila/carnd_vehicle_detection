import numpy as np
import cv2

SMOOTHING_KERNEL = np.array([[0.25, 0.25], [0.25, 0.25]])

class AggregatedHeatmap:
    def __init__(self):
        self.smoothed_heatmaps = np.zeros((5, 720, 1280))

    def process_new_heatmap(self, heatmap):
        self.smoothed_heatmaps = np.roll(self.smoothed_heatmaps, 1, 0)
        self.smoothed_heatmaps[0] = self.smooth_heatmap(heatmap)


    @staticmethod
    def smooth_heatmap(heatmap):
        return cv2.filter2D(heatmap, -1, SMOOTHING_KERNEL)

    def smoothed_heatmap(self):
        return np.average(self.smoothed_heatmaps, 0, [30, 28, 14, 10, 8])
               # * np.apply_along_axis(np.count_nonzero, 0, self.smoothed_heatmaps)
