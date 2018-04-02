#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import sys

import h5py
import numpy as np
from sample_collection_helper import feature_label_concatenation_h5py
from sample_collection_helper import simple_sample_weighting
from sample_collection_helper import get_onset_in_frame_helper
from sample_collection_helper import feature_onset_phrase_label_sample_weights

sys.path.append(os.path.join(os.path.dirname(__file__), "../src/"))

from parameters_jingju import *
from file_path_jingju_shared import *

from textgridParser import textGrid2WordList
from textgridParser import wordListsParseByLines

from audio_preprocessing import getMFCCBands2DMadmom
from sklearn import preprocessing

from utilFunctions import featureReshape
from utilFunctions import getRecordings
from schluterParser import annotationCvParser


# jingju annotation format ---------------------------------------------------------------------------------------------
def dump_training_data_textgrid_helper(wav_path,
                                       textgrid_path,
                                       recording_name,
                                       tier_parent=None,
                                       tier_child=None):
    """
    load audio, textgrid
    :param wav_path:
    :param textgrid_path:
    :param artist_name:
    :param recording_name:
    :param tier_parent: parent tier can be line
    :param tier_childL child tier can be syllable
    :return:
    """

    ground_truth_textgrid_file = os.path.join(textgrid_path, recording_name + '.TextGrid')
    wav_file = os.path.join(wav_path, recording_name + '.wav')
    line_list = textGrid2WordList(ground_truth_textgrid_file, whichTier=tier_parent)
    utterance_list = textGrid2WordList(ground_truth_textgrid_file, whichTier=tier_child)

    # parse lines of groundtruth
    nested_utterance_lists, num_lines, num_utterances = wordListsParseByLines(line_list, utterance_list)

    # load audio
    log_mel = getMFCCBands2DMadmom(wav_file, fs, hopsize_t, channel=1)

    return nested_utterance_lists, log_mel


def dump_feature_sample_weights_onset(wav_path,
                                      textgrid_path,
                                      recordings,
                                      tier_parent=None,
                                      tier_child=None):
    """
    as the function name
    :param wav_path:
    :param textgrid_path:
    :param score_path:
    :param recordings:
    :param tier_parent:
    :param tier_child:
    :return:
    """
    log_mel_p_all = []
    log_mel_n_all = []
    sample_weights_p_all = []
    sample_weights_n_all = []

    for recording_name in recordings:

        nested_utterance_lists, log_mel = dump_training_data_textgrid_helper(wav_path=wav_path,
                                                                             textgrid_path=textgrid_path,
                                                                             recording_name=recording_name,
                                                                             tier_parent=tier_parent,
                                                                             tier_child=tier_child)

        # create the ground truth lab files
        for idx, u_list in enumerate(nested_utterance_lists):

            if not len(u_list[1]):
                continue

            frames_onset, frame_start, frame_end = \
                get_onset_in_frame_helper(recording_name, idx, lab=False, u_list=u_list)

            log_mel_p, log_mel_n, sample_weights_p, sample_weights_n = \
                simple_sample_weighting(log_mel, frames_onset, frame_start, frame_end)

            log_mel_p_all.append(log_mel_p)
            log_mel_n_all.append(log_mel_n)
            sample_weights_p_all.append(sample_weights_p)
            sample_weights_n_all.append(sample_weights_n)

    return np.concatenate(log_mel_p_all), \
           np.concatenate(log_mel_n_all), \
           np.concatenate(sample_weights_p_all), \
           np.concatenate(sample_weights_n_all)


def dump_feature_label_sample_weights(path_audio,
                                      path_textgrid,
                                      path_output=None,
                                      tier_parent=None,
                                      tier_child=None):
    """
    dump features for all the dataset for onset detection
    :return:
    """

    recording_names = getRecordings(path_textgrid)

    log_mel_p, \
    log_mel_n, \
    sample_weights_p, \
    sample_weights_n \
        = dump_feature_sample_weights_onset(wav_path=path_audio,
                                            textgrid_path=path_textgrid,
                                            recordings=recording_names,
                                            tier_parent=tier_parent,
                                            tier_child=tier_child)

    print('finished feature extraction.')

    sample_weights = np.concatenate((sample_weights_p, sample_weights_n))

    # save h5py separately for postive and negative features
    filename_log_mel_p = join(path_output, 'log_mel_p.h5')
    h5f = h5py.File(filename_log_mel_p, 'w')
    h5f.create_dataset('mfcc_p', data=log_mel_p)
    h5f.close()

    filename_log_mel_n = join(feature_data_path, 'log_mel_n.h5')
    h5f = h5py.File(filename_log_mel_n, 'w')
    h5f.create_dataset('mfcc_n', data=log_mel_n)
    h5f.close()

    del log_mel_p
    del log_mel_n

    feature_all, label_all, scaler = \
        feature_label_concatenation_h5py(filename_log_mel_p,
                                         filename_log_mel_n,
                                         scaling=True)

    print('finished feature concatenation.')

    os.remove(filename_log_mel_p)
    os.remove(filename_log_mel_n)

    feature_all = featureReshape(feature_all, nlen=7)

    print('feature shape:', feature_all.shape)

    # save feature
    filename_feature_all = join(path_output, 'feature.h5')
    h5f = h5py.File(filename_feature_all, 'w')
    h5f.create_dataset('feature_all', data=feature_all)
    h5f.close()

    # save label, sample weights and scaler
    pickle.dump(label_all, open(join(path_output, 'labels.pkl'), 'wb'), protocol=2)

    pickle.dump(sample_weights, open(join(path_output, 'sample_weights.pkl'), 'wb'), protocol=2)

    pickle.dump(scaler, open(join(path_output, 'scaler.pkl'), 'wb'), protocol=2)


def save_feature_label_sample_weights_onset_phrase(wav_path,
                                                   textgrid_path,
                                                   path_output,
                                                   tier_parent,
                                                   tier_child):
    """
    phrase level feature and sample weights dump
    :param wav_path:
    :param textgrid_path:
    :param recordings:
    :param tier_parent:
    :param tier_child
    :return:
    """
    recordings = getRecordings(textgrid_path)

    log_mel_line_all = []
    for recording_name in recordings:

        nested_utterance_lists, log_mel = dump_training_data_textgrid_helper(wav_path=wav_path,
                                                                             textgrid_path=textgrid_path,
                                                                             recording_name=recording_name,
                                                                             tier_parent=tier_parent,
                                                                             tier_child=tier_child)
        # create the ground truth lab files
        for idx, u_list in enumerate(nested_utterance_lists):
            print 'Processing feature collecting ... ' + recording_name + ' phrase ' + str(idx + 1)

            if not len(u_list[1]):
                continue

            frames_onset, frame_start, frame_end = \
                get_onset_in_frame_helper(recording_name, idx, lab=False, u_list=u_list)

            log_mel_line, label, sample_weights = \
                feature_onset_phrase_label_sample_weights(frames_onset, frame_start, frame_end, log_mel)

            # save feature in h5py
            filename_log_mel_line = join(path_output, 'feature' + '_' +recording_name+'_'+str(idx)+'.h5')
            h5f = h5py.File(filename_log_mel_line, 'w')
            h5f.create_dataset('feature_all', data=log_mel_line)
            h5f.close()

            # dumpy label
            filename_label = join(path_output, 'label' + '_' +recording_name+'_'+str(idx)+'.pkl')
            pickle.dump(label, open(filename_label, 'wb'), protocol=2)

            # dump sample weights
            filename_sample_weights = join(path_output, 'sample_weights' + '_' +
                                           recording_name + '_' + str(idx) + '.pkl')
            pickle.dump(sample_weights, open(filename_sample_weights, 'wb'), protocol=2)

            log_mel_line_all.append(log_mel_line)

    log_mel_line_all = np.concatenate(log_mel_line_all)

    # dump scaler
    scaler = preprocessing.StandardScaler()

    scaler.fit(log_mel_line_all)

    filename_scaler = join(path_output, 'scaler_phrase.pkl')
    pickle.dump(scaler, open(filename_scaler, 'wb'))

    return True


# bock annotation format -----------------------------------------------------------------------------------------------
def feature_label_weights_saver(path_output, filename, feature, label, sample_weights):

    filename_feature_all = join(path_output, 'feature_' + filename + '.h5')
    h5f = h5py.File(filename_feature_all, 'w')
    h5f.create_dataset('feature_all', data=feature)
    h5f.close()

    pickle.dump(label, open(join(path_output, 'label_' + filename + '.pkl'), 'wb'), protocol=2)

    pickle.dump(sample_weights, open(join(path_output, 'sample_weights_' + filename + '.pkl'), 'wb'), protocol=2)


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


def dump_feature_label_sample_weights_onset_phrase_bock(audio_path,
                                                        annotation_path,
                                                        path_output):
    """
    dump feature, label, sample weights for each phrase with bock annotation format
    :param audio_path:
    :param annotation_path:
    :param path_output:
    :return:
    """

    log_mel_line_all = []
    for fn in getRecordings(annotation_path):

        # from the annotation to get feature, frame start and frame end of each line, frames_onset
        log_mel, frames_onset, frame_start, frame_end = dump_feature_onset_helper(audio_path, annotation_path, fn, 1)

        # simple sample weighting
        feature, label, sample_weights = \
            feature_onset_phrase_label_sample_weights(frames_onset, frame_start, frame_end, log_mel)

        # save feature, label and weights
        feature_label_weights_saver(path_output, fn, feature, label, sample_weights)

        log_mel_line_all.append(feature)

    log_mel_line_all = np.concatenate(log_mel_line_all)

    scaler = preprocessing.StandardScaler()

    scaler.fit(log_mel_line_all)

    pickle.dump(scaler, open(path_output, 'wb'))

    return True


if __name__ == '__main__':
    import argparse

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description="dump feature, label and sample weights for general purpose.")
    parser.add_argument("--audio",
                        type=str,
                        help="input audio path")
    parser.add_argument("--annotation",
                        type=str,
                        help="input annotation path")
    parser.add_argument("--output",
                        type=str,
                        help="output path")
    parser.add_argument("-at",
                        "--annotation_type",
                        type=str,
                        default='jingju',
                        choices=['jingju', 'bock'],
                        help="annotation types, please choose from jingju or bock")
    parser.add_argument("-p",
                        "--phrase",
                        type=str2bool,
                        default='false',
                        help="whether to collect feature for each phrase")
    parser.add_argument("-tp",
                        "--tier_parent",
                        type=str,
                        default="line",
                        help="Parent tier of the textgrid annotation")
    parser.add_argument("-tc",
                        "--tier_child",
                        type=str,
                        default="dianSilence",
                        help="Child tier of the textgrid annotation")
    args = parser.parse_args()

    if not os.path.isdir(args.audio):
        raise OSError('Audio path %s not found' % args.audio)

    if not os.path.isdir(args.annotation):
        raise OSError('Annotation path %s not found' % args.annotation)

    if not os.path.isdir(args.output):
        raise OSError('Output path %s not found' % args.output)

    if args.annotation_type == 'bock':
        dump_feature_label_sample_weights_onset_phrase_bock(audio_path=args.audio,
                                                            annotation_path=args.annotation,
                                                            path_output=args.output)
    else:
        if args.phrase:
            save_feature_label_sample_weights_onset_phrase(wav_path=args.audio,
                                                           textgrid_path=args.annotation,
                                                           path_output=args.output,
                                                           tier_parent=args.tier_parent,
                                                           tier_child=args.tier_child)
        else:
            dump_feature_label_sample_weights(path_audio=args.audio,
                                              path_textgrid=args.annotation,
                                              path_output=args.output,
                                              tier_parent=args.tier_parent,
                                              tier_child=args.tier_child)
