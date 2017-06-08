import os
import pickle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from carnd_vehicle_detection import ROOT_DIR
from carnd_vehicle_detection.preprocess import extract_features, read_training_data

DEFAULT_CLASSIFIER_SAVE_PATH = os.path.join(ROOT_DIR, 'svm_classifier.p')

# FIXME: how to structure the training and validation data sets for training the classifier?


class DummyScaler:
    def transform(self, x):
        return x


my_dummy_scaler = DummyScaler()

def get_classifier(classifier_path=None, classifier_save_path=DEFAULT_CLASSIFIER_SAVE_PATH,
                   features_train=None, labels_train=None,
                   features_valid=None, labels_valid=None):
    """Read the vehicle / non-vehicle classifier, based on whether a non-None path is provided or not.
    
    :param classifier_path: The path of a previously trained classifier. Can be none, in which case the classifier is 
           trained from scratch using the projec't default training set. 
    :param classifier_save_path: The path where the newly trained classifier is to be saved. Used only if no 
           classifier_path is set           
    :param features_train: Training features as an numpy ndarray
    :param labels_train: Training labels as an numpy ndarray
    :param features_valid: Validation features as an numpy ndarray
    :param labels_valid: Validation labels as an numpy ndarray
    :returns: A classifier object. 
    """

    # It is all or nothing baby. If any of these is missing, just use the defaults.

    # FIXME, this is just a placeholder

    if classifier_path is not None:
        with open(classifier_path, 'rb') as classifier_file:
            classifier = pickle.load(classifier_file)
        score = None
        scaler = None
    else:
        if None in (features_train, labels_train, features_valid, labels_valid):
            features_train, features_valid, labels_train, labels_valid = read_training_data()
        extracted_features_train = extract_features(features_train)
        extracted_features_valid = extract_features(features_valid)
        scaler = StandardScaler()
        scaler.fit(extracted_features_train)
        scaled_features_train = scaler.transform(extracted_features_train)
        scaled_features_valid = scaler.transform(extracted_features_valid)

        classifier, score = train_classifier(scaled_features_train, labels_train, scaled_features_valid, labels_valid)
        if classifier_save_path is not None:
            with open(classifier_save_path, 'wb') as outfile:
                pickle.dump(classifier, outfile)

    return {'classifier': classifier, 'score': score, 'scaler': scaler}


def train_classifier(features_train, labels_train, features_valid, labels_valid, output_path=DEFAULT_CLASSIFIER_SAVE_PATH):
    """FIXME document the method. What kinds of features are acceptable etc.
    :param features_train: The training features, FIXME
    :param labels_train: The training labels, a numpy ndarray of same length as features_train, containing zeros 
           and ones
    :param features_valid: The validationfeatures, same requirements as for features_train
    :param labels_valid: The validation labels, a numpy ndarray of same length as features_valid, containing zeros 
           and ones
    :return: A two-tuple containing the classifier object and the accuracy score (float)"""

    svc = LinearSVC()
    svc.fit(features_train, labels_train)
    pred = svc.predict(features_valid)
    score = accuracy_score(labels_valid, pred)
    print("Successfully trained the classifier. Accuracy on test set was {}".format(score))
    return svc, score


