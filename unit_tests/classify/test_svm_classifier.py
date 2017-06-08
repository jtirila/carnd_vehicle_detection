from carnd_vehicle_detection.classify.svm_classifier import get_classifier, DEFAULT_CLASSIFIER_SAVE_PATH
from unit_tests.preprocessing.test_read_classification_training_data import \
    VEHICLES_TEST_IMAGE_EXPR, NONVEHICLES_TEST_IMAGE_EXPR, read_training_data
import unittest
from sklearn.svm import LinearSVC


class TestSvmClassifier(unittest.TestCase):
    def test_initializing_classifier_with_training_data_provided(self):
        features_train, features_valid, labels_train, labels_valid = \
            read_training_data(VEHICLES_TEST_IMAGE_EXPR, NONVEHICLES_TEST_IMAGE_EXPR)
        clf_and_score = get_classifier(None, DEFAULT_CLASSIFIER_SAVE_PATH, features_train, labels_train, features_valid, labels_valid)
        clf = clf_and_score['classifier']
        score = clf_and_score['score']
        self.assertTrue(isinstance(clf, LinearSVC))
        self.assertEquals(score, 1.0)
