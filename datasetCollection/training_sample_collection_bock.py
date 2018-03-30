#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import h5py
import pickle
import cPickle
import gzip
import os
import sys

from sklearn import preprocessing
from sample_collection_helper import feature_label_concatenation
from sample_collection_helper import complicate_sample_weighting
from sample_collection_helper import positive_three_sample_weighting
from sample_collection_helper import simple_sample_weighting

sys.path.append(os.path.join(os.path.dirname(__file__), "../src/"))

from parameters_schluter import *
from file_path_bock import *

from schluterParser import annotationCvParser
from utilFunctions import getRecordings
from audio_preprocessing import getMFCCBands2DMadmom
from sample_collection_helper import feature_onset_phrase_label_sample_weights

# import essentia.standard as ess
# from training_sample_collection_jingju import getMFCCBands2D
# from training_sample_collection_jingju import feature_label_concatenation


def dump_feature_onset_helper(audio_path, annotation_path, fn, channel):

    audio_fn = join(audio_path, fn + '.flac')
    annotation_fn = join(annotation_path, fn + '.onsets')

    mfcc = getMFCCBands2DMadmom(audio_fn, fs, hopsize_t, channel)

    print('Feature collecting ...', fn)

    times_onset = annotationCvParser(annotation_fn)
    times_onset = [float(to) for to in times_onset]
    # syllable onset frames
    frames_onset = np.array(np.around(np.array(times_onset) / hopsize_t), dtype=int)

    # line start and end frames
    frame_start = 0
    frame_end = mfcc.shape[0] - 1

    return mfcc, frames_onset, frame_start, frame_end


def feature_data_path_switcher(sampleWeighting, channel):

    if sampleWeighting == 'complicate':
        feature_path = schluter_feature_data_path_madmom_complicateSampleWeighting
    elif sampleWeighting == 'positiveThree':
        feature_path = schluter_feature_data_path_madmom_positiveThreeSampleWeighting
    else:
        if channel == 1:
            feature_path = schluter_feature_data_path_madmom_simpleSampleWeighting
        else:
            feature_path = schluter_feature_data_path_madmom_simpleSampleWeighting_3channel

    return feature_path


def feature_label_weights_saver(feature_path, feature_all, label_all, sample_weights):

    filename_feature_all = join(feature_path, 'feature_' + fn + '.h5')
    h5f = h5py.File(filename_feature_all, 'w')
    h5f.create_dataset('feature_all', data=feature_all)
    h5f.close()

    cPickle.dump(label_all,
                 gzip.open(join(feature_path, 'label_' + fn + '.pickle.gz'), 'wb'), protocol=2)

    cPickle.dump(sample_weights,
                 gzip.open(join(feature_path, 'sample_weights_' + fn + '.pickle.gz'), 'wb'), protocol=2)


def dump_feature_onset(audio_path,
                       annotation_path,
                       fn,
                       channel=1,
                       sampleWeighting='complicate'):
    """
    dump feature, label, sample weight for onsets
    :param audio_path:
    :param annotation_path:
    :param fn:
    :param channel:
    :param sampleWeighting:
    :return:
    """

    mfcc, frames_onset, frame_start, frame_end = \
        dump_feature_onset_helper(audio_path=audio_path,
                                  annotation_path=annotation_path,
                                  fn=fn,
                                  channel=channel)

    if sampleWeighting == 'complicate':
        print('complicate weighting...')
        mfcc_p, mfcc_n, sample_weights_p, sample_weights_n = \
            complicate_sample_weighting(mfcc, frames_onset, frame_start, frame_end)
    elif sampleWeighting == 'positiveThree':
        print('positive three weighting...')
        mfcc_p, mfcc_n, sample_weights_p, sample_weights_n = \
            positive_three_sample_weighting(mfcc, frames_onset, frame_start, frame_end)
    else:
        print('simple weighting...')
        mfcc_p, mfcc_n, sample_weights_p, sample_weights_n = \
            simple_sample_weighting(mfcc, frames_onset, frame_start, frame_end)

    feature_all, label_all, scaler = feature_label_concatenation(mfcc_p, mfcc_n, scaling=False)

    sample_weights = np.concatenate((sample_weights_p, sample_weights_n))

    feature_path = feature_data_path_switcher(sampleWeighting, channel)

    feature_label_weights_saver(feature_path, feature_all, label_all, sample_weights)


def dump_feature_onset_phrase(audio_path,
                              annotation_path,
                              fn,
                              channel=1):
    """
    dump feature, label, sample weights for each phrase
    :param audio_path:
    :param annotation_path:
    :param fn:
    :param channel:
    :return:
    """

    # from the annotation to get feature, frame start and frame end of each line, frames_onset
    mfcc, frames_onset, frame_start, frame_end = dump_feature_onset_helper(audio_path, annotation_path, fn, channel)

    # simple sample weighting
    mfcc_line, label, sample_weights = \
        feature_onset_phrase_label_sample_weights(frames_onset, frame_start, frame_end, mfcc)

    feature_path = schluter_feature_data_path_madmom_simpleSampleWeighting_phrase

    if not os.path.exists(feature_path):
        os.makedirs(feature_path)

    # save feature, label and weights
    feature_label_weights_saver(feature_path, mfcc_line, label, sample_weights)

    return mfcc_line

if __name__ == '__main__':
    import argparse

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description="dump feature, label and sample weights for bock train set.")
    parser.add_argument("-p",
                        "--phrase",
                        type=str2bool,
                        default='true',
                        help="whether to collect feature for each phrase")
    args = parser.parse_args()

    if args.phrase:
        mfcc_line_all = []
        for fn in getRecordings(schluter_annotations_path):
            mfcc_line = dump_feature_onset_phrase(audio_path=schluter_audio_path,
                                                  annotation_path=schluter_annotations_path,
                                                  fn=fn,
                                                  channel=1)
            mfcc_line_all.append(mfcc_line)

        mfcc_line_all = np.concatenate(mfcc_line_all)

        scaler = preprocessing.StandardScaler()

        scaler.fit(mfcc_line_all)

        pickle.dump(scaler, open(scaler_schluter_phrase_model_path, 'wb'))
    # else:
    #     for fn in getRecordings(schluter_annotations_path):
    #         dump_feature_onset(audio_path=schluter_audio_path,
    #                            annotation_path=schluter_annotations_path,
    #                            fn=fn,
    #                            channel=1,
    #                            sampleWeighting='simple')
