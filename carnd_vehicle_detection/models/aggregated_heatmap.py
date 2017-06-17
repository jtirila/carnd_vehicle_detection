import numpy as np
from multiprocessing import Pool
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
        p = Pool(4)
        arr = self.smoothed_heatmaps[:, :, :]
        slice1 = arr[:, :360, :640]
        slice2 = arr[:, 360:, :640]
        slice3 = arr[:, :360, 640:]
        slice4 = arr[:, 360:, 640:]
        slices = p.map(self.some_fun, [slice1, slice2, slice3, slice4])
        return_arr = np.zeros((arr.shape[1], arr.shape[2]))
        return_arr[:360, :640] = slices[0]
        return_arr[360:, :640] = slices[1]
        return_arr[:360, 640:] = slices[2]
        return_arr[360:, 640:] = slices[3]
        return return_arr

    @staticmethod
    def some_fun(x):
        return np.average(x, 0, [50, 40, 30, 25, 25, 20, 20]) * np.apply_along_axis(np.count_nonzero, 0, x)
