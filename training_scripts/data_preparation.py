import pickle
import numpy as np
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split


def load_data_jingju(filename_labels_train_validation_set,
                     filename_sample_weights):

    # load training and validation data

    with open(filename_labels_train_validation_set, 'r') as f:
        Y_train_validation = pickle.load(f)

    with open(filename_sample_weights, 'r') as f:
        sample_weights = pickle.load(f)

    print('negative sample size:', len(Y_train_validation[Y_train_validation == 0]),
          'postive sample size:', len(Y_train_validation[Y_train_validation == 1]))

    indices_features = range(len(Y_train_validation))

    class_weights = compute_class_weight('balanced',[0,1],Y_train_validation)

    class_weights = {0:class_weights[0], 1:class_weights[1]}

    filenames_features_sample_weights = \
        np.array([[indices_features[ii], sample_weights[ii]] for ii in range(len(indices_features))])

    filenames_sample_weights_train, filenames_sample_weights_validation, Y_train, Y_validation = \
        train_test_split(filenames_features_sample_weights,
                         Y_train_validation,
                         test_size=0.1,
                         stratify=Y_train_validation)

    indices_train = [int(xt[0]) for xt in filenames_sample_weights_train]
    sample_weights_train = np.array([xt[1] for xt in filenames_sample_weights_train])
    indices_validation = [int(xv[0]) for xv in filenames_sample_weights_validation]
    sample_weights_validation = np.array([xv[1] for xv in filenames_sample_weights_validation])

    return indices_train, Y_train, sample_weights_train, \
           indices_validation, Y_validation, sample_weights_validation, \
           indices_features, Y_train_validation, sample_weights, class_weights


def load_data_bock(filename_labels_train_validation_set,
                   filename_sample_weights):

    # load training and validation data

    with open(filename_labels_train_validation_set, 'rb') as f:
        Y_train_validation = pickle.load(f)

    with open(filename_sample_weights, 'rb') as f:
        sample_weights = pickle.load(f)

    print(len(Y_train_validation[Y_train_validation == 0]), len(Y_train_validation[Y_train_validation == 1]))

    # this is the filename indices
    indices_train_validation = range(len(Y_train_validation))

    class_weights = compute_class_weight('balanced', [0, 1], Y_train_validation)

    class_weights = {0: class_weights[0], 1: class_weights[1]}

    return indices_train_validation, Y_train_validation, sample_weights, class_weights