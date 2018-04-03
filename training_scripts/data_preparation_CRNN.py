import h5py
import pickle
import numpy as np
import csv
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from utilFunctions import featureReshape
from utilFunctions import append_or_write


def featureLabelSampleWeightsLoad(data_path, filename, scaler):

    # load data
    mfcc_line_path = os.path.join(data_path, 'feature' + '_' + filename + '.h5')
    label_path = os.path.join(data_path, 'label' + '_' + filename + '.pkl')
    sample_weights_path = os.path.join(data_path, 'sample_weights' + '_' + filename + '.pkl')

    f = h5py.File(mfcc_line_path, 'r')
    mfcc_line = f['feature_all']
    label = pickle.load(open(label_path, 'r'))
    sample_weights = pickle.load(open(sample_weights_path, 'r'))

    # scaling feature
    mfcc_line = scaler.transform(mfcc_line)

    return mfcc_line, label, sample_weights


def featureLabelSampleWeightsPad(mfcc_line, label, sample_weights, len_seq):
    """
    load and pad feature label and sample weights
    :param data_path:
    :param filename:
    :param scaler:
    :param len_seq: sequence length
    :return:
    """

    # length of the paded sequence
    len_2_pad = int(len_seq * np.ceil(len(mfcc_line)/float(len_seq)))

    mfcc_line_pad, label_pad, sample_weights_pad, len_padded = \
        featureLabelSampleWeightsPad2Length(mfcc_line, label, sample_weights, len_2_pad)

    return mfcc_line_pad, label_pad, sample_weights_pad, len_padded


def featureLabelSampleWeightsPad2Length(mfcc_line, label, sample_weights, len_2_pad):
    """
    pad them to the len_2_pad length
    :param mfcc_line:
    :param label:
    :param sample_weights:
    :param len_2_pad:
    :return:
    """
    len_padded = len_2_pad - len(mfcc_line)

    # pad feature, label and sample weights
    mfcc_line_pad = np.zeros((len_2_pad, mfcc_line.shape[1]), dtype='float32')
    mfcc_line_pad[:mfcc_line.shape[0], :] = mfcc_line
    mfcc_line_pad = featureReshape(mfcc_line_pad, nlen=7)

    label_pad = np.zeros((len_2_pad,), dtype='int')
    label_pad[:len(label)] = label

    sample_weights_pad = np.zeros((len_2_pad,), dtype='float32')
    sample_weights_pad[:len(sample_weights)] = sample_weights

    return mfcc_line_pad, label_pad, sample_weights_pad, len_padded


def reinitialize_train_variables():
    list_mfcc_line, list_label, list_sample_weights = [], [], []
    len_max_train = 0
    batch_counter = -1
    return list_mfcc_line, list_label, list_sample_weights, len_max_train, batch_counter


def reinitialize_train_tensor(input_shape, batch_size, len_seq):
    mfcc_line_tensor = np.zeros(input_shape, dtype='float32')
    label_tensor = np.zeros((batch_size, len_seq, 1), dtype='int')
    sample_weights_tensor = np.zeros((batch_size, len_seq))
    return mfcc_line_tensor, label_tensor, sample_weights_tensor


def createInputTensor(mfcc_line_pad, label_pad, sample_weights_pad, len_seq, ii):
    """
    segment input into subsequences
    :param mfcc_line_pad:
    :param label_pad:
    :param sample_weights_pad:
    :param len_seq:
    :param ii: sub sequence index
    :return:
    """
    mfcc_line_tensor = mfcc_line_pad[ii * len_seq:(ii + 1) * len_seq]
    mfcc_line_tensor = np.expand_dims(mfcc_line_tensor, axis=0)
    mfcc_line_tensor = np.expand_dims(mfcc_line_tensor, axis=2)

    label_tensor = label_pad[ii * len_seq:(ii + 1) * len_seq]
    label_tensor = np.expand_dims(label_tensor, axis=0)
    label_tensor = np.expand_dims(label_tensor, axis=2)

    sample_weights_tensor = sample_weights_pad[ii * len_seq:(ii + 1) * len_seq]
    sample_weights_tensor = np.expand_dims(sample_weights_tensor, axis=0)

    return mfcc_line_tensor, label_tensor, sample_weights_tensor


def writeValLossCsv(file_path_log, ii_epoch, val_loss, train_loss=None):
    """
    write epoch number and validation loss to csv file
    :param file_path_log:
    :param ii_epoch:
    :param val_loss:
    :return:
    """
    append_write = append_or_write(file_path_log)
    with open(file_path_log, append_write) as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        if train_loss is not None:
            writer.writerow([ii_epoch, train_loss, val_loss])
        else:
            writer.writerow([ii_epoch, val_loss])