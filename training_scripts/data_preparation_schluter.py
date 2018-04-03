import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from schluterParser import annotationCvParser
from file_path_bock import *
from parameters_jingju import *
from utilFunctions import getRecordings
from utilFunctions import featureReshape
from utilFunctions import featureDereshape
from sklearn import preprocessing
import gzip, cPickle
from os.path import isfile
import numpy as np
import h5py

def getTrainingFilenames(annotation_path, cv_filename):
    """
    annotation filenames - cv test filenames
    :param annotation_path:
    :param cv_filename:
    :return:
    """
    annotation_fns = getRecordings(annotation_path)
    test_fns = annotationCvParser(cv_filename)
    train_fns = [x for x in annotation_fns if x not in test_fns]
    return train_fns

def concatenateFeatureLabelSampleweights(train_fns,
                                         schluter_feature_data_path,
                                         n_pattern=21,
                                         nlen=10,
                                         scaling=True,
                                         channel=1):
    """
    concatenate feature label and sample weights
    :param train_fns:
    :return:
    """
    label_all = []
    sample_weights_all = []
    for fn in train_fns:
        sample_weights_fn = join(schluter_feature_data_path, 'sample_weights_' + fn + '.pickle.gz')
        label_fn = join(schluter_feature_data_path, 'label_'+fn+'.pickle.gz')

        if not isfile(sample_weights_fn):
            print(sample_weights_fn, 'not found.')
            continue
        if not isfile(label_fn):
            print(label_fn, 'not found.')
            continue


        with gzip.open(sample_weights_fn, 'rb') as f:
            sample_weights = cPickle.load(f)
            sample_weights_all.append(sample_weights)

        with gzip.open(label_fn, 'rb') as f:
            label = cPickle.load(f)
            label_all.append(label)

    sample_weights_all = np.concatenate(sample_weights_all)
    label_all = np.concatenate(label_all)
    print(label_all)

    nDims = 80*n_pattern
    if channel == 1:
        feature_all = np.zeros((len(label_all), nDims), dtype='float32')
    else:
        feature_all = np.zeros((len(label_all), nDims, 3), dtype='float32')

    idx_start = 0
    for fn in train_fns:
        # print('Concatenating feature ...', fn, 'idx start', idx_start)
        feature_fn = join(schluter_feature_data_path, 'feature_'+fn+'.h5')
        if not isfile(feature_fn):
            print(feature_fn, 'not found.')
            continue
        feature = h5py.File(feature_fn, 'r')
        dim_feature = feature['feature_all'].shape[0]
        if channel == 1:
            feature_all[idx_start:(idx_start + dim_feature), :] = feature['feature_all']
        else:
            feature_all[idx_start:(idx_start + dim_feature), :, :] = feature['feature_all']
        idx_start += dim_feature
        feature.flush()
        feature.close()

    if scaling:
        if channel == 1:
            scaler = preprocessing.StandardScaler()
            scaler.fit(feature_all)
            feature_all = scaler.transform(feature_all)
        else:
            scaler = []
            for ii in range(3):
                scaler_temp = preprocessing.StandardScaler()
                scaler_temp.fit(feature_all[:, :, ii])
                feature_all[:,:,ii] = scaler_temp.transform(feature_all[:,:, ii])
                scaler.append(scaler_temp)
    else:
        scaler = None

    if channel == 1:
        feature_all = featureReshape(feature_all, nlen=nlen)
    else:
        feature_all_conc = []
        for ii in range(3):
            feature_all_conc.append(featureReshape(feature_all[:,:,ii], nlen=nlen))
        feature_all = np.stack(feature_all_conc, axis=3)

    return feature_all, label_all, sample_weights_all, scaler


def concatenateFeatureLabelSampleweightsJingju(feature_schluter,
                                               label_schluter,
                                               sample_weights_schluter,
                                               filename_jingju_features,
                                               filename_jingju_labels,
                                               filename_jingju_sample_weights,
                                               nlen=7,
                                               scaling=True):
    """
    Concatenate bock and jingju dataset
    :param feature_schluter:
    :param label_schluter:
    :param sample_weights_schluter:
    :param filename_jingju_features:
    :param filename_jingju_labels:
    :param filename_jingju_sample_weights:
    :param scaling:
    :return:
    """
    # load jingju feature, labels and sample weights
    feature_jingju = h5py.File(filename_jingju_features, 'r')

    with gzip.open(filename_jingju_labels, 'rb') as f:
        label_jingju = cPickle.load(f)

    with gzip.open(filename_jingju_sample_weights, 'rb') as f:
        sample_weights_jingju = cPickle.load(f)

    # concatenate with bock dataset
    feature_all = np.vstack((feature_schluter, feature_jingju['feature_all']))
    label_all = np.concatenate((label_schluter, label_jingju))
    sample_weights_all = np.concatenate((sample_weights_schluter, sample_weights_jingju))

    feature_all = featureDereshape(feature_all, nlen=nlen)

    if scaling:
        scaler = preprocessing.StandardScaler()
        scaler.fit(feature_all)
        feature_all = scaler.transform(feature_all)
    else:
        scaler = None

    feature_all = featureReshape(feature_all, nlen=nlen)

    return feature_all, label_all, sample_weights_all, scaler


def saveFeatureLabelSampleweights(feature_all, label_all, sample_weights, scaler,
                                  feature_fn, label_fn, sample_weights_fn, scaler_fn):
    h5f = h5py.File(feature_fn, 'w')
    h5f.create_dataset('feature_all', data=feature_all)
    h5f.close()

    cPickle.dump(label_all,
                 gzip.open(
                     label_fn,
                     'wb'), cPickle.HIGHEST_PROTOCOL)

    cPickle.dump(sample_weights,
                 gzip.open(
                     sample_weights_fn,
                     'wb'), cPickle.HIGHEST_PROTOCOL)

    cPickle.dump(scaler,
                 gzip.open(
                     scaler_fn,
                     'wb'), cPickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    for ii in range(1,8):
        test_cv_filename = join(bock_cv_path, '8-fold_cv_random_' + str(ii) + '.fold')
        train_fns = getTrainingFilenames(bock_annotations_path, test_cv_filename)
        print(len(train_fns))
        feature_all, label_all, sample_weights_all, scaler = \
            concatenateFeatureLabelSampleweights(train_fns,
                                                 bock_feature_data_path_madmom_simpleSampleWeighting,
                                                 n_pattern=15,
                                                 nlen=7,
                                                 scaling=True)
        print(feature_all.shape)
        print(len(label_all))
        print(len(sample_weights_all))
