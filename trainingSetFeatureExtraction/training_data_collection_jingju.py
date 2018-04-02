#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import sys

import h5py
import numpy as np
from sklearn import preprocessing

from sample_collection_helper import complicate_sample_weighting
from sample_collection_helper import feature_label_concatenation
from sample_collection_helper import feature_label_concatenation_h5py
from sample_collection_helper import positive_three_sample_weighting
from sample_collection_helper import simple_sample_weighting
from sample_collection_helper import feature_onset_phrase_label_sample_weights
from sample_collection_helper import get_onset_in_frame_helper

sys.path.append(os.path.join(os.path.dirname(__file__), "../src/"))

from parameters_jingju import *
from file_path_jingju_shared import *

from textgridParser import textGrid2WordList
from textgridParser import wordListsParseByLines
from scoreParser import csvDurationScoreParser
from trainTestSeparation import getTestTrainRecordingsNactaISMIR
from trainTestSeparation import getTestTrainRecordingsArtistAlbumFilter

from audio_preprocessing import getMFCCBands2DMadmom

from labParser import lab2WordList
from utilFunctions import featureReshape

# from Fdeltas import Fdeltas
# from Fprev_sub import Fprev_sub
# from audio_preprocessing import _nbf_2D
# from textgridParser import syllableTextgridExtraction
# from sklearn.model_selection import train_test_split
# import essentia.standard as ess


def dump_feature_onset_helper(lab,
                              wav_path,
                              textgrid_path,
                              score_path,
                              artist_name,
                              recording_name,
                              feature_type):
    """
    load or parse audio, textgrid
    :param lab:
    :param wav_path:
    :param textgrid_path:
    :param score_path:
    :param artist_name:
    :param recording_name:
    :param feature_type:
    :return:
    """
    if not lab:
        ground_truth_textgrid_file = os.path.join(textgrid_path, artist_name, recording_name + '.TextGrid')
        wav_file = os.path.join(wav_path, artist_name, recording_name + '.wav')
        line_list = textGrid2WordList(ground_truth_textgrid_file, whichTier='line')
        utterance_list = textGrid2WordList(ground_truth_textgrid_file, whichTier='dianSilence')

        # parse lines of groundtruth
        nested_utterance_lists, num_lines, num_utterances = wordListsParseByLines(line_list, utterance_list)
    else:
        ground_truth_textgrid_file = os.path.join(textgrid_path, artist_name, recording_name + '.lab')
        wav_file = os.path.join(wav_path, artist_name, recording_name + '.mp3')
        nested_utterance_lists = [lab2WordList(ground_truth_textgrid_file, label=True)]

    # parse score
    score_file = os.path.join(score_path, artist_name, recording_name + '.csv')
    _, utterance_durations, bpm = csvDurationScoreParser(score_file)

    # load audio
    if feature_type == 'madmom':
        mfcc = getMFCCBands2DMadmom(wav_file, fs, hopsize_t, channel=1)
    else:
        print(feature_type + ' is not exist.')
        raise

    return nested_utterance_lists, utterance_durations, bpm, mfcc


def dump_feature_sample_weights_onset(wav_path,
                                      textgrid_path,
                                      score_path,
                                      recordings,
                                      feature_type='mfcc',
                                      lab=False,
                                      sampleWeighting='simple'):
    """
    as the function name
    :param wav_path:
    :param textgrid_path:
    :param score_path:
    :param recordings:
    :param feature_type:
    :param lab:
    :param sampleWeighting:
    :return:
    """

    # p: position, n: negative, 75: 0.75 sample_weight
    mfcc_p_all = []
    mfcc_n_all = []
    sample_weights_p_all = []
    sample_weights_n_all = []

    for artist_name, recording_name in recordings:

        nested_utterance_lists, utterance_durations, bpm, mfcc = dump_feature_onset_helper(lab,
                                                                                           wav_path,
                                                                                           textgrid_path,
                                                                                           score_path,
                                                                                           artist_name,
                                                                                           recording_name,
                                                                                           feature_type)

        # create the ground truth lab files
        for idx, u_list in enumerate(nested_utterance_lists):
            try:
                print("beat per minute:", bpm[idx])
            except IndexError:
                continue

            if float(bpm[idx]):

                frames_onset, frame_start, frame_end = get_onset_in_frame_helper(recording_name, idx, lab, u_list)

                if sampleWeighting == 'complicate':
                    print('complicate weighting')
                    mfcc_p, mfcc_n, sample_weights_p, sample_weights_n = \
                        complicate_sample_weighting(mfcc, frames_onset, frame_start, frame_end)
                elif sampleWeighting == 'positiveThree':
                    print('postive three weighting')
                    mfcc_p, mfcc_n, sample_weights_p, sample_weights_n = \
                        positive_three_sample_weighting(mfcc, frames_onset, frame_start, frame_end)
                else:
                    mfcc_p, mfcc_n, sample_weights_p, sample_weights_n = \
                        simple_sample_weighting(mfcc, frames_onset, frame_start, frame_end)

                mfcc_p_all.append(mfcc_p)
                mfcc_n_all.append(mfcc_n)
                sample_weights_p_all.append(sample_weights_p)
                sample_weights_n_all.append(sample_weights_n)

    return np.concatenate(mfcc_p_all), \
           np.concatenate(mfcc_n_all), \
           np.concatenate(sample_weights_p_all), \
           np.concatenate(sample_weights_n_all)


def save_feature_label_sample_weights_onset_phrase(wav_path,
                                                   textgrid_path,
                                                   score_path,
                                                   recordings,
                                                   feature_type='mfcc',
                                                   lab=False,
                                                   split='ismir'):
    """
    phrase level feature and sample weights dump
    :param wav_path:
    :param textgrid_path:
    :param score_path:
    :param recordings:
    :param feature_type:
    :param lab:
    :param split:
    :return:
    """
    mfcc_line_all = []
    for artist_name, recording_name in recordings:

        nested_utterance_lists, utterance_durations, bpm, mfcc = dump_feature_onset_helper(lab,
                                                                                           wav_path,
                                                                                           textgrid_path,
                                                                                           score_path,
                                                                                           artist_name,
                                                                                           recording_name,
                                                                                           feature_type)

        # create the ground truth lab files
        for idx, u_list in enumerate(nested_utterance_lists):
            try:
                print(bpm[idx])
            except IndexError:
                continue

            if float(bpm[idx]):
                print 'Processing feature collecting ... ' + recording_name + ' phrase ' + str(idx + 1)

                frames_onset, frame_start, frame_end = get_onset_in_frame_helper(recording_name, idx, lab, u_list)

                mfcc_line, label, sample_weights = \
                    feature_onset_phrase_label_sample_weights(frames_onset, frame_start, frame_end, mfcc)

                # save feature, label, sample weights
                feature_data_split_path = join(feature_data_path, 'jingju_phrase')
                if not os.path.exists(feature_data_split_path):
                    os.makedirs(feature_data_split_path)

                # save feature in h5py
                filename_mfcc_line = join(feature_data_split_path,
                                          'feature'+'_'+artist_name + '_' +recording_name+'_'+str(idx)+'.h5')
                h5f = h5py.File(filename_mfcc_line, 'w')
                h5f.create_dataset('feature_all', data=mfcc_line)
                h5f.close()

                # dumpy label in pickle.gz
                filename_label = join(feature_data_split_path,
                                      'label'+'_'+artist_name + '_' +recording_name+'_'+str(idx)+'.pkl')
                pickle.dump(label, open(filename_label, 'wb'), protocol=2)

                # dump sample weights in pickle.gz
                filename_sample_weights = join(feature_data_split_path,
                                               'sample_weights' + '_' + artist_name + '_' +
                                               recording_name + '_' + str(idx) + '.pkl')
                pickle.dump(sample_weights, open(filename_sample_weights, 'wb'), protocol=2)

                mfcc_line_all.append(mfcc_line)

    return np.concatenate(mfcc_line_all)


def dataset_switcher(split='ismir'):
    if split =='ismir':
        testNacta2017, testNacta, trainNacta2017, trainNacta = getTestTrainRecordingsNactaISMIR()
    elif split == 'artist_filter':
        testNacta2017, testNacta, trainNacta2017, trainNacta = getTestTrainRecordingsArtistAlbumFilter()
    return testNacta2017, testNacta, trainNacta2017, trainNacta


def dump_feature_batch_onset(split='artist_filter',
                             train_test ='train',
                             feature_type='madmom',
                             sampleWeighting='simple'):
    """
    dump features for all the dataset for onset detection
    :return:
    """

    testNacta2017, testNacta, trainNacta2017, trainNacta = dataset_switcher(split)

    if train_test == 'train':
        nacta_data = trainNacta
        nacta_data_2017 = trainNacta2017
        scaling = True
    else:
        nacta_data = testNacta
        nacta_data_2017 = testNacta2017
        scaling = False

    if len(trainNacta2017):
        mfcc_p_nacta2017, \
        mfcc_n_nacta2017, \
        sample_weights_p_nacta2017, \
        sample_weights_n_nacta2017 \
            = dump_feature_sample_weights_onset(wav_path=nacta2017_wav_path,
                                                textgrid_path=nacta2017_textgrid_path,
                                                score_path=nacta2017_score_path,
                                                recordings=nacta_data_2017,
                                                feature_type=feature_type,
                                                sampleWeighting=sampleWeighting)

    mfcc_p_nacta, \
    mfcc_n_nacta, \
    sample_weights_p_nacta, \
    sample_weights_n_nacta \
        = dump_feature_sample_weights_onset(wav_path=nacta_wav_path,
                                            textgrid_path=nacta_textgrid_path,
                                            score_path=nacta_score_path,
                                            recordings=nacta_data,
                                            feature_type=feature_type,
                                            sampleWeighting=sampleWeighting)

    print('finished feature extraction.')

    if len(trainNacta2017):
        mfcc_p = np.concatenate((mfcc_p_nacta2017, mfcc_p_nacta))
        mfcc_n = np.concatenate((mfcc_n_nacta2017, mfcc_n_nacta))
        sample_weights_p = np.concatenate((sample_weights_p_nacta2017, sample_weights_p_nacta))
        sample_weights_n = np.concatenate((sample_weights_n_nacta2017, sample_weights_n_nacta))
    else:
        mfcc_p = mfcc_p_nacta
        mfcc_n = mfcc_n_nacta
        sample_weights_p = sample_weights_p_nacta
        sample_weights_n = sample_weights_n_nacta

    sample_weights = np.concatenate((sample_weights_p, sample_weights_n))

    # save h5py separately for postive and negative features
    filename_mfcc_p = join(feature_data_path, 'mfcc_p_' + split + '_split.h5')
    h5f = h5py.File(filename_mfcc_p, 'w')
    h5f.create_dataset('mfcc_p', data=mfcc_p)
    h5f.close()

    filename_mfcc_n = join(feature_data_path, 'mfcc_n_' + split + '_split.h5')
    h5f = h5py.File(filename_mfcc_n, 'w')
    h5f.create_dataset('mfcc_n', data=mfcc_n)
    h5f.close()

    del mfcc_p
    del mfcc_n

    feature_all, label_all, scaler = feature_label_concatenation_h5py(filename_mfcc_p, filename_mfcc_n, scaling=scaling)

    os.remove(filename_mfcc_p)
    os.remove(filename_mfcc_n)

    if train_test == 'train':
        if feature_type != 'madmom':
            nlen = 10
        else:
            nlen = 7

        feature_all = featureReshape(feature_all, nlen=nlen)

    print('feature shape:', feature_all.shape)

    filename_feature_all = join(feature_data_path,
                                'feature_all_'+split+'_'+feature_type+'_'+train_test+'_'+sampleWeighting+'.h5')
    h5f = h5py.File(filename_feature_all, 'w')
    h5f.create_dataset('feature_all', data=feature_all)
    h5f.close()

    print('finished feature concatenation.')

    pickle.dump(label_all, open(join(feature_data_path, 'labels.pkl'), 'wb'), protocol=2)

    if train_test == 'train':
        pickle.dump(sample_weights, open(join(feature_data_path, 'sample_weights.pkl'),'wb'), protocol=2)

        pickle.dump(scaler, open(join(feature_data_path, 'scaler.pkl'), 'wb'), protocol=2)


def dump_feature_batch_onset_phrase(split='ismir', feature_type='mfccBands2D', train_test='train'):
    """
    dump features for each phrase (line)
    :return:
    """

    testNacta2017, testNacta, trainNacta2017, trainNacta = dataset_switcher(split)

    if train_test == 'train':
        nacta_data = trainNacta
        nacta_data_2017 = trainNacta2017
    else:
        nacta_data = testNacta
        nacta_data_2017 = testNacta2017

    if len(trainNacta2017):

        mfcc_line_nacta2017 = save_feature_label_sample_weights_onset_phrase(wav_path=nacta2017_wav_path,
                                                                             textgrid_path=nacta2017_textgrid_path,
                                                                             score_path=nacta2017_score_path,
                                                                             recordings=nacta_data_2017,
                                                                             feature_type=feature_type,
                                                                             split=split)

    mfcc_line_nacta = save_feature_label_sample_weights_onset_phrase(wav_path=nacta_wav_path,
                                                                     textgrid_path=nacta_textgrid_path,
                                                                     score_path=nacta_score_path,
                                                                     recordings=nacta_data,
                                                                     feature_type=feature_type,
                                                                     split=split)

    print('finished feature extraction.')

    if len(trainNacta2017):
        mfcc_all = np.concatenate((mfcc_line_nacta2017, mfcc_line_nacta))
    else:
        mfcc_all = mfcc_line_nacta

    if train_test == 'train':
        scaler = preprocessing.StandardScaler()
        scaler.fit(mfcc_all)

        filename_scaler = join(feature_data_path, 'scaler_jingju_phrase'+'.pkl')

        pickle.dump(scaler, open(filename_scaler, 'wb'))


def dump_feature_batch_onset_test():
    """
    dump features for the test dataset for onset detection
    :return:
    """
    testNacta2017, testNacta, trainNacta2017, trainNacta = getTestTrainRecordingsArtistAlbumFilter()

    mfcc_p_nacta, \
    mfcc_n_nacta, \
    sample_weights_p_nacta, \
    sample_weights_n_nacta \
        = dump_feature_sample_weights_onset(wav_path=nacta_wav_path,
                                            textgrid_path=nacta_textgrid_path,
                                            score_path=nacta_score_path,
                                            recordings=testNacta,
                                            feature_type='mfccBands2D')

    mfcc_p_nacta1017, \
    mfcc_n_nacta2017, \
    sample_weights_p_nacta2017, \
    sample_weights_n_nacta2017 \
        = dump_feature_sample_weights_onset(wav_path=nacta2017_wav_path,
                                            textgrid_path=nacta2017_textgrid_path,
                                            score_path=nacta2017_score_path,
                                            recordings=testNacta2017,
                                            feature_type='mfccBands2D')

    print('finished feature extraction.')

    mfcc_p = np.concatenate((mfcc_p_nacta1017, mfcc_p_nacta))
    mfcc_n = np.concatenate((mfcc_n_nacta2017, mfcc_n_nacta))

    print('finished feature concatenation.')

    feature_all, label_all, scaler = feature_label_concatenation(mfcc_p, mfcc_n, scaling=False)

    print(mfcc_p.shape, mfcc_n.shape)

    h5f = h5py.File(join(feature_data_path, 'feature_test_set.h5', 'w'))
    h5f.create_dataset('feature_all', data=feature_all)
    h5f.close()

    pickle.dump(label_all, open(join(feature_data_path, 'label_test_set.pkl'), 'wb'), protocol=2)


if __name__ == '__main__':
    import argparse

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description="dump feature, label and sample weights for jingju train set.")
    parser.add_argument("-p",
                        "--phrase",
                        type=str2bool,
                        default='false',
                        help="whether to collect feature for each phrase")
    args = parser.parse_args()

    if args.phrase:
        dump_feature_batch_onset_phrase(split='artist_filter',
                                        feature_type='madmom',
                                        train_test='train')
    else:
        dump_feature_batch_onset(split='artist_filter',
                                 feature_type='madmom',
                                 train_test='train',
                                 sampleWeighting='simple')
