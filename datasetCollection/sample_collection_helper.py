import os
import sys
import h5py
import numpy as np
from sklearn import preprocessing

sys.path.append(os.path.join(os.path.dirname(__file__), "../src/"))

from phonemeMap import dic_pho_label


def feature_label_concatenation(mfcc_p, mfcc_n, scaling=False):
    """
    organize the training feature and label
    :param
    :return:
    """

    label_p = [1] * mfcc_p.shape[0]
    label_n = [0] * mfcc_n.shape[0]

    feature_all = np.concatenate((mfcc_p, mfcc_n), axis=0)
    label_all = label_p+label_n

    feature_all = np.array(feature_all, dtype='float32')
    label_all = np.array(label_all, dtype='int64')

    scaler = None

    return feature_all, label_all, scaler


def feature_label_concatenation_h5py(filename_mfcc_p, filename_mfcc_n, scaling=True):
    """
    organize the training feature and label
    :param
    :return:
    """
    # open h5py
    f_mfcc_p = h5py.File(filename_mfcc_p, 'a')
    f_mfcc_n = h5py.File(filename_mfcc_n, 'r')

    # get the feature shape
    dim_p_0 = f_mfcc_p['mfcc_p'].shape[0]
    dim_n_0 = f_mfcc_n['mfcc_n'].shape[0]
    dim_1 = f_mfcc_p['mfcc_p'].shape[1]

    # concatenate labels
    label_p = [1] * dim_p_0
    label_n = [0] * dim_n_0
    label_all = label_p + label_n

    feature_all = np.zeros((dim_p_0+dim_n_0, dim_1), dtype='float32')

    print('concatenate features... ...')

    feature_all[:dim_p_0, :] = f_mfcc_p['mfcc_p']
    feature_all[dim_p_0:, :] = f_mfcc_n['mfcc_n']

    # free the memory of the h5py
    f_mfcc_p.flush()
    f_mfcc_p.close()
    f_mfcc_n.flush()
    f_mfcc_n.close()

    label_all = np.array(label_all, dtype='int64')

    print('scaling features... ... ')

    # scaling
    scaler = preprocessing.StandardScaler()
    scaler.fit(feature_all)
    if scaling:
        feature_all = scaler.transform(feature_all)

    return feature_all, label_all, scaler


def feature_label_concatenation_phoneme(dic_pho_feature_train):
    """
    organize the training feature and label
    :param dic_pho_feature_train: input dictionary, key: phoneme, value: feature vectors
    :return:
    """
    feature_all = []
    label_all = []
    for key in dic_pho_feature_train:
        feature = dic_pho_feature_train[key]
        label = [dic_pho_label[key]] * len(feature)

        if len(feature):
            if not len(feature_all):
                feature_all = feature
            else:
                feature_all = np.vstack((feature_all, feature))
            label_all += label
    label_all = np.array(label_all, dtype='int64')

    scaler = preprocessing.StandardScaler()
    scaler.fit(feature_all)
    feature_all = scaler.transform(feature_all)

    return feature_all, label_all, scaler


def remove_out_of_range(frames, frame_start, frame_end):
    return frames[np.all([frames <= frame_end, frames >= frame_start], axis=0)]


def complicate_sample_weighting(mfcc, frames_onset, frame_start, frame_end):
    """
    Weight +/-6 frames around the onset frame, first 3 frames as the positive weighting.
    :param mfcc:
    :param frames_onset:
    :param frame_start:
    :param frame_end:
    :return:
    """
    frames_onset_p75 = np.hstack((frames_onset - 1, frames_onset + 1))
    frames_onset_p50 = np.hstack((frames_onset - 2, frames_onset + 2))
    frames_onset_p25 = np.hstack((frames_onset - 3, frames_onset + 3))

    frames_onset_p75 = remove_out_of_range(frames_onset_p75, frame_start, frame_end)
    frames_onset_p50 = remove_out_of_range(frames_onset_p50, frame_start, frame_end)
    frames_onset_p25 = remove_out_of_range(frames_onset_p25, frame_start, frame_end)

    # mfcc positive
    mfcc_p100 = mfcc[frames_onset, :]
    mfcc_p75 = mfcc[frames_onset_p75, :]
    mfcc_p50 = mfcc[frames_onset_p50, :]
    mfcc_p25 = mfcc[frames_onset_p25, :]

    frames_n25 = np.hstack((frames_onset - 4, frames_onset + 4))
    frames_n50 = np.hstack((frames_onset - 5, frames_onset + 5))
    frames_n75 = np.hstack((frames_onset - 6, frames_onset + 6))

    frames_n25 = remove_out_of_range(frames_n25, frame_start, frame_end)
    frames_n50 = remove_out_of_range(frames_n50, frame_start, frame_end)
    frames_n75 = remove_out_of_range(frames_n75, frame_start, frame_end)

    # mfcc negative
    mfcc_n25 = mfcc[frames_n25, :]
    mfcc_n50 = mfcc[frames_n50, :]
    mfcc_n75 = mfcc[frames_n75, :]

    frames_all = np.arange(frame_start, frame_end)
    frames_n100 = np.setdiff1d(frames_all, np.hstack((frames_onset,
                                                      frames_onset_p75,
                                                      frames_onset_p50,
                                                      frames_onset_p25,
                                                      frames_n25,
                                                      frames_n50,
                                                      frames_n75)))
    mfcc_n100 = mfcc[frames_n100, :]

    mfcc_p = np.concatenate((mfcc_p100, mfcc_p75, mfcc_p50, mfcc_p25), axis=0)
    sample_weights_p = np.concatenate((np.ones((mfcc_p100.shape[0],)),
                                       np.ones((mfcc_p75.shape[0],)) * 0.75,
                                       np.ones((mfcc_p50.shape[0],)) * 0.5,
                                       np.ones((mfcc_p25.shape[0],)) * 0.25))

    mfcc_n = np.concatenate((mfcc_n100, mfcc_n75, mfcc_n50, mfcc_n25), axis=0)
    sample_weights_n = np.concatenate((np.ones((mfcc_n100.shape[0],)),
                                       np.ones((mfcc_n75.shape[0],)) * 0.75,
                                       np.ones((mfcc_n50.shape[0],)) * 0.5,
                                       np.ones((mfcc_n25.shape[0],)) * 0.25))

    return mfcc_p, mfcc_n, sample_weights_p, sample_weights_n


def positive_three_sample_weighting(mfcc, frames_onset, frame_start, frame_end):
    """
    Weight +/-3 frames around the onset frame as positive weights.
    :param mfcc:
    :param frames_onset:
    :param frame_start:
    :param frame_end:
    :return:
    """
    frames_onset_p75 = np.hstack((frames_onset - 1, frames_onset + 1))
    frames_onset_p50 = np.hstack((frames_onset - 2, frames_onset + 2))
    frames_onset_p25 = np.hstack((frames_onset - 3, frames_onset + 3))

    frames_onset_p75 = remove_out_of_range(frames_onset_p75, frame_start, frame_end)
    frames_onset_p50 = remove_out_of_range(frames_onset_p50, frame_start, frame_end)
    frames_onset_p25 = remove_out_of_range(frames_onset_p25, frame_start, frame_end)

    # mfcc positive
    mfcc_p100 = mfcc[frames_onset, :]
    mfcc_p75 = mfcc[frames_onset_p75, :]
    mfcc_p50 = mfcc[frames_onset_p50, :]
    mfcc_p25 = mfcc[frames_onset_p25, :]

    frames_all = np.arange(frame_start, frame_end)
    frames_n100 = np.setdiff1d(frames_all, np.hstack((frames_onset,
                                                      frames_onset_p75,
                                                      frames_onset_p50,
                                                      frames_onset_p25)))
    mfcc_n100 = mfcc[frames_n100, :]

    mfcc_p = np.concatenate((mfcc_p100, mfcc_p75, mfcc_p50, mfcc_p25), axis=0)
    sample_weights_p = np.concatenate((np.ones((mfcc_p100.shape[0],)),
                                       np.ones((mfcc_p75.shape[0],)) * 0.75,
                                       np.ones((mfcc_p50.shape[0],)) * 0.5,
                                       np.ones((mfcc_p25.shape[0],)) * 0.25))

    mfcc_n = mfcc_n100
    sample_weights_n = np.ones((mfcc_n100.shape[0],))

    return mfcc_p, mfcc_n, sample_weights_p, sample_weights_n


def simple_sample_weighting(mfcc, frames_onset, frame_start, frame_end):
    """
    simple weighing strategy used in Schluter's paper
    :param mfcc:
    :param frames_onset:
    :param frame_start:
    :param frame_end:
    :return:
    """

    frames_onset_p25 = np.hstack((frames_onset - 1, frames_onset + 1))
    frames_onset_p25 = remove_out_of_range(frames_onset_p25, frame_start, frame_end)

    # mfcc positive
    mfcc_p100 = mfcc[frames_onset, :]
    mfcc_p25 = mfcc[frames_onset_p25, :]

    frames_all = np.arange(frame_start, frame_end)
    frames_n100 = np.setdiff1d(frames_all, np.hstack((frames_onset,
                                                      frames_onset_p25)))

    mfcc_n100 = mfcc[frames_n100, :]

    mfcc_p = np.concatenate((mfcc_p100, mfcc_p25), axis=0)
    sample_weights_p = np.concatenate((np.ones((mfcc_p100.shape[0],)),
                                       np.ones((mfcc_p25.shape[0],)) * 0.25))

    mfcc_n = mfcc_n100
    sample_weights_n = np.ones((mfcc_n100.shape[0],))

    return mfcc_p, mfcc_n, sample_weights_p, sample_weights_n


def feature_onset_phrase_label_sample_weights(frames_onset, frame_start, frame_end, mfcc):
    frames_onset_p25 = np.hstack((frames_onset - 1, frames_onset + 1))
    frames_onset_p25 = remove_out_of_range(frames_onset_p25, frame_start, frame_end)

    len_line = frame_end - frame_start + 1

    mfcc_line = mfcc[frame_start:frame_end+1, :]

    sample_weights = np.ones((len_line,))
    sample_weights[frames_onset_p25 - frame_start] = 0.25

    label = np.zeros((len_line,))
    label[frames_onset - frame_start] = 1
    label[frames_onset_p25 - frame_start] = 1

    return mfcc_line, label, sample_weights
