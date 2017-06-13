import os
import pickle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from carnd_vehicle_detection import ROOT_DIR
from carnd_vehicle_detection.preprocess import single_img_features, read_training_data, normalize_luminosity

DEFAULT_CLASSIFIER_SAVE_PATH = os.path.join(ROOT_DIR, 'svm_classifier.p')


def get_classifier(classifier_path=None, classifier_save_path=DEFAULT_CLASSIFIER_SAVE_PATH,
                   features_train=None, labels_train=None,
                   features_valid=None, labels_valid=None, extract_features_dict=None):
    """Read the vehicle / non-vehicle classifier, based on whether a non-None path is provided or not.
    
    :param classifier_path: The path of a previously trained classifier. Can be none, in which case the classifier is 
           trained from scratch using the projec't default training set. 
    :param classifier_save_path: The path where the newly trained classifier is to be saved. Used only if no 
           classifier_path is set           
    :param features_train: Training features as an numpy ndarray
    :param labels_train: Training labels as an numpy ndarray
    :param features_valid: Validation features as an numpy ndarray
    :param labels_valid: Validation labels as an numpy ndarray
    :param extract_features_dict: The extract parameters in a dict. See the extract_features module for details
    :return: A dict with keys 'classifier', 'score' and 'scaler', and corresponding values.
    """

    assert extract_features_dict is not None

    # It is all or nothing baby. If any of these is missing, just use the defaults.

    if classifier_path is not None:
        with open(classifier_path, 'rb') as classifier_file:
            classifier = pickle.load(classifier_file)
        with open(".".join(classifier_save_path.split(".")[:-1]) + '_scaler.p', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        score = None
    else:
        if None in (features_train, labels_train, features_valid, labels_valid):
            features_train, features_valid, labels_train, labels_valid = read_training_data()
        extracted_features_train = [single_img_features(normalize_luminosity(img), **extract_features_dict)
                                    for img in features_train]
        extracted_features_valid = [single_img_features(normalize_luminosity(img), **extract_features_dict)
                                    for img in features_valid]
        scaler = StandardScaler()
        scaler.fit(extracted_features_train)
        scaled_features_train = scaler.transform(extracted_features_train)
        scaled_features_valid = scaler.transform(extracted_features_valid)

        classifier, score = train_classifier(scaled_features_train, labels_train, scaled_features_valid, labels_valid)
        if classifier_save_path is not None:
            with open(classifier_save_path, 'wb') as outfile:
                pickle.dump(classifier, outfile)
            with open(".".join(classifier_save_path.split(".")[:-1]) + '_scaler.p', 'wb') as scalerfile:
                pickle.dump(scaler, scalerfile)

    return {'classifier': classifier, 'score': score, 'scaler': scaler}


def train_classifier(features_train, labels_train, features_valid, labels_valid):
    """Train a Linear SVM classifier based on the data provided as input. 
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


