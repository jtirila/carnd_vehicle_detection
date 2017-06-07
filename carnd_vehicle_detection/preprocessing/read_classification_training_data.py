import os
from carnd_vehicle_detection import ROOT_DIR
from sklearn.model_selection import train_test_split
from glob import glob
import numpy as np
import matplotlib.image as mpimg

DEFAULT_VEHICLE_IMAGE_SEARCH_EXPRESSION = os.path.join(ROOT_DIR, 'images', 'vehicles', "**/*png")
DEFAULT_NONVEHICLE_IMAGE_SEARCH_EXPRESSION = os.path.join(ROOT_DIR, 'images', 'nonvehicles', "**/*png")


def read_training_data(vehicles_search_expr=DEFAULT_VEHICLE_IMAGE_SEARCH_EXPRESSION,
                       nonvehicles_search_expr=DEFAULT_NONVEHICLE_IMAGE_SEARCH_EXPRESSION):
    """FIXME: may need to be divided into separate functions for training, validation, test data sets. Depends."""
    vehicle_paths = glob(vehicles_search_expr, recursive=True)
    vehicles = np.array([mpimg.imread(fname) for fname in vehicle_paths])
    vehicle_labels = np.full(vehicles.shape[:2], 1)
    nonvehicle_paths = glob(nonvehicles_search_expr, recursive=True)
    nonvehicles = np.array([mpimg.imread(fname) for fname in nonvehicle_paths])
    nonvehicle_labels = np.full(nonvehicles.shape[:2], 0)
    features = np.vstack((vehicles, nonvehicles))
    labels = np.vstack((vehicle_labels, nonvehicle_labels))
    return train_test_split(features, labels, test_size=0.33, random_state=np.random.randint(1000))
