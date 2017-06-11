import numpy as np

class DetectedVehicle:
    """Instances of this class will be used to keep track of high-probability vehicle detections. An instance 
    corresponds to an area in a frame that has multiple detections in a constrained area. Furthermore, evidence is 
    collected about appearance in a sequence of frames instead of just one frame, and this information will be 
    used in reasoning about the confidence of the detection.
     """
    def __init__(self, windows):
        self.previous_bounding_boxes = [None] * 10
        self.bounding_box = None
        self.windows = windows
        self.bounding_box = self._calculate_bounding_box()

    def _center_of_bounding_box(self):
        hori = 0.5 * (self._left_x_coordinate_of_bounding_box(self.bounding_box) +
                      self._right_x_coordinate_of_bounding_box(self.bounding_box))
        vert = 0.5 * (self._top_y_coordinate_of_bounding_box(self.bounding_box) +
                      self._bottom_y_coordinate_of_bounding_box(self.bounding_box))
        return vert, hori

    def _calculate_projected_bounding_box(self):
        if None in self.previous_bounding_boxes[:3]:
            return self.previous_bounding_boxes[0]
        else:
            try:
                stop_ind = self.previous_bounding_boxes.index(None)
            except ValueError:
                # None not in list
                stop_ind = len(self.previous_bounding_boxes)
            inds = list(range(stop_ind)[::-1])
            left = self._polynomial_projection(inds, self.previous_bounding_boxes[:stop_ind, 0, 0])
            right = self._polynomial_projection(inds, self.previous_bounding_boxes[:stop_ind, 1, 0])


    @staticmethod
    def _polynomial_projection(previous_indices, previous_values):
        coeffs = np.polyfit(previous_indices, previous_values, 2)
        new_ind = np.max(previous_indices) + 1
        return coeffs[0] * new_ind ** 2 + coeffs[1] * new_ind + coeffs[0]

    def _calculate_bounding_box(self):
        left = np.min(np.ravel(self.windows[:, :, 0]))
        right = np.max(np.ravel(self.windows[:, :, 0]))
        top = np.min(np.ravel(self.windows[:, :, 1]))
        bottom = np.max(np.ravel(self.windows[:, :, 1]))
        return (left, top), (right, bottom)

    # Some convenience methods for bounding box calculations
    def _left_x_coordinate_of_bounding_box(self):
        return self.bounding_box[0][0]

    def _right_x_coordinate_of_bounding_box(self):
        return self.bounding_box[1][0]

    def _top_y_coordinate_of_bounding_box(self):
        return self.bounding_box[0][1]

    def _bottom_y_coordinate_of_bounding_box(self):
        return self.bounding_box[1][1]

    @staticmethod
    def partition_windows(windows, detected_vehicles):
        """FIXME: Partition the collection of car-detection windows into disjoint sets."""
        pass

    @staticmethod
    def windows_overlap(win1, win2):
        """
        :param win1:  ((x11, y11), (x12, y12))
        :param win2:  ((x21, y21), (x22, y22)) 
        :return: 
        """
        for win1_tmp, win2_tmp in ((win1, win2), (win2, win1)):
            win1left, win2right = win1_tmp[0][0], win2_tmp[1][0]
            if win1left >= win2right:
                return False
            win1top, win2bottom = win1_tmp[0][1], win2_tmp[1][1]
            if win1top >= win2bottom:
                return False
        return True
