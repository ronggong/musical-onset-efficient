# -*- coding: utf-8 -*-
from os import makedirs
from os.path import isfile
from os.path import exists
import gzip
import pickle
import cPickle
import numpy as np
import pyximport

from keras.models import load_model
from src.file_path_jingju_no_rnn import *
from src.parameters_jingju import *
from src.labWriter import boundaryLabWriter
from src.utilFunctions import featureReshape
from src.utilFunctions import smooth_obs
from src.trainTestSeparation import getTestRecordingsScoreDurCorrectionArtistAlbumFilter
from experiment_process_helper import data_parser
from experiment_process_helper import get_line_properties
from experiment_process_helper import boundary_decoding
from experiment_process_helper import get_results_decoding_path
from experiment_process_helper import get_boundary_list
from experiment_process_helper import write_results_2_txt_jingju
from experiment_process_helper import odf_calculation_no_crnn

from eval_demo import eval_write_2_txt
from madmom.features.onsets import OnsetPeakPickingProcessor
from audio_preprocessing import getMFCCBands2DMadmom
from plot_code import plot_jingju

pyximport.install(reload_support=True,
                  setup_args={'include_dirs': np.get_include()})

import viterbiDecoding


def batch_process_onset_detection(wav_path,
                                  textgrid_path,
                                  score_path,
                                  scaler,
                                  test_recordings,
                                  model_keras_cnn_0,
                                  cnnModel_name,
                                  eval_results_path,
                                  architecture=varin['architecture'],
                                  lab=False,
                                  threshold=0.54,
                                  obs_cal=True,
                                  decoding_method='viterbi'):
    """
    :param wav_path: string, path where we have the audio files
    :param textgrid_path:  string, path where we have the text grid ground truth
    :param score_path: string, path where we have the scores
    :param scaler: scaler object sklearn
    :param test_recordings: list of strings, test recording filenames
    :param model_keras_cnn_0: keras .h5, CNN onset detection model
    :param cnnModel_name: string, CNN model name
    :param eval_results_path: string, path where we save the evaluation results
    :param architecture: string, the model architecture
    :param lab: bool, used for Riyaz dataset
    :param threshold: float, used for peak picking
    :param obs_cal: bool, if to calculate the ODF or not
    :param decoding_method: string, viterbi or peakPicking
    :return:
    """

    eval_results_decoding_path = \
        get_results_decoding_path(decoding_method=decoding_method,
                                  bool_corrected_score_duration=varin['corrected_score_duration'],
                                  eval_results_path=eval_results_path)

    # loop through all recordings
    for artist_path, rn in test_recordings:

        score_file = join(score_path, artist_path, rn+'.csv')

        if not isfile(score_file):
            print('Score not found: ' + score_file)
            continue

        nested_syllable_lists, wav_file, line_list, syllables, syllable_durations, bpm, pinyins = \
            data_parser(artist_path=artist_path,
                        wav_path=wav_path,
                        textgrid_path=textgrid_path,
                        rn=rn,
                        score_file=score_file,
                        lab=lab)

        if obs_cal == 'tocal':
            # load audio
            mfcc = getMFCCBands2DMadmom(wav_file, fs, hopsize_t, channel=1)
            mfcc_scaled = scaler.transform(mfcc)
            mfcc_reshaped = featureReshape(mfcc_scaled, nlen=7)

        i_line = -1
        for i_obs, line in enumerate(line_list):
            # line without lyrics will be ignored
            if not lab and len(line[2]) == 0:
                continue

            i_line += 1

            # line without duration will be ignored
            try:
                print(syllable_durations[i_line])
            except IndexError:
                continue

            # line non-fixed tempo will be ignored
            if float(bpm[i_line]) == 0:
                continue

            time_line, lyrics_line, frame_start, frame_end = get_line_properties(lab=lab,
                                                                                 line=line,
                                                                                 hopsize_t=hopsize_t)

            # initialize ODF path and filename
            obs_path = join('./obs', cnnModel_name, artist_path)
            obs_filename = rn + '_' + str(i_line + 1) + '.pkl'

            if obs_cal == 'tocal':

                obs_i, mfcc_line = odf_calculation_no_crnn(mfcc=mfcc,
                                                           mfcc_reshaped=mfcc_reshaped,
                                                           filename_keras_cnn_0=filename_keras_cnn_0,
                                                           model_keras_cnn_0=model_keras_cnn_0,
                                                           architecture=architecture,
                                                           frame_start=frame_start,
                                                           frame_end=frame_end)

                # save onset curve
                print('save onset curve ... ...')
                if not exists(obs_path):
                    makedirs(obs_path)
                pickle.dump(obs_i, open(join(obs_path, obs_filename), 'w'))
            else:
                obs_i = pickle.load(open(join(obs_path, obs_filename), 'r'))

            obs_i = np.squeeze(obs_i)
            obs_i = smooth_obs(obs_i)

            # organize score
            print('Calculating: ', rn, ' phrase ', str(i_obs))
            print('ODF Methods: ', architecture)

            # process the score duration
            duration_score = syllable_durations[i_line]
            # only save the duration if it exists
            duration_score = np.array([float(ds) for ds in duration_score if len(ds)])
            # normalize the duration
            duration_score *= (time_line/np.sum(duration_score))

            i_boundary, label = boundary_decoding(decoding_method=decoding_method,
                                                  obs_i=obs_i,
                                                  duration_score=duration_score,
                                                  varin=varin,
                                                  threshold=threshold,
                                                  hopsize_t=hopsize_t,
                                                  viterbiDecoding=viterbiDecoding,
                                                  OnsetPeakPickingProcessor=OnsetPeakPickingProcessor)

            # create detected syllable result filename
            filename_syll_lab = join(eval_results_decoding_path, artist_path,
                                     rn + '_' + str(i_line + 1) + '.syll.lab')
            time_boundary_start = np.array(i_boundary[:-1]) * hopsize_t
            time_boundary_end = np.array(i_boundary[1:]) * hopsize_t

            boundary_list = get_boundary_list(lab=lab,
                                              decoding_method=decoding_method,
                                              time_boundary_start=time_boundary_start,
                                              time_boundary_end=time_boundary_end,
                                              pinyins=pinyins,
                                              syllables=syllables,
                                              i_line=i_line)

            boundaryLabWriter(boundaryList=boundary_list,
                              outputFilename=filename_syll_lab,
                              label=label)

            if varin['plot'] and obs_cal == 'tocal':
                plot_jingju(nested_syllable_lists=nested_syllable_lists,
                            i_line=i_line,
                            mfcc_line=mfcc_line,
                            hopsize_t=hopsize_t,
                            obs_i=obs_i,
                            i_boundary=i_boundary,
                            duration_score=duration_score)

    return eval_results_decoding_path


def viterbi_subroutine(test_nacta_2017, test_nacta, eval_label, obs_cal):
    """5 run times routine for the viterbi decoding onset detection"""

    list_recall_onset_25, list_precision_onset_25, list_F1_onset_25 = [], [], []
    list_recall_onset_5, list_precision_onset_5, list_F1_onset_5 = [], [], []
    list_recall_25, list_precision_25, list_F1_25 = [], [], []
    list_recall_5, list_precision_5, list_F1_5 = [], [], []
    for ii in range(5):

        if obs_cal == 'tocal':

            if 'schluter' in varin['architecture']:
                scaler = cPickle.load(gzip.open(full_path_mfccBands_2D_scaler_onset+str(ii)+'.pickle.gz'))

            model_keras_cnn_0 = load_model(full_path_keras_cnn_0 + str(ii) + '.h5')
            # print(model_keras_cnn_0.summary())
            print('Model name:', full_path_keras_cnn_0)

            if varin['dataset'] != 'ismir':
                # nacta2017
                batch_process_onset_detection(wav_path=nacta2017_wav_path,
                                              textgrid_path=nacta2017_textgrid_path,
                                              score_path=nacta2017_score_unified_path,
                                              test_recordings=test_nacta_2017,
                                              model_keras_cnn_0=model_keras_cnn_0,
                                              cnnModel_name=cnnModel_name + str(ii),
                                              eval_results_path=eval_results_path + str(ii),
                                              scaler=scaler,
                                              architecture=varin['architecture'],
                                              obs_cal=obs_cal,
                                              decoding_method='viterbi')

            # nacta
            eval_results_decoding_path = batch_process_onset_detection(wav_path=nacta_wav_path,
                                                                       textgrid_path=nacta_textgrid_path,
                                                                       score_path=nacta_score_unified_path,
                                                                       test_recordings=test_nacta,
                                                                       model_keras_cnn_0=model_keras_cnn_0,
                                                                       cnnModel_name=cnnModel_name + str(ii),
                                                                       eval_results_path=eval_results_path + str(ii),
                                                                       scaler=scaler,
                                                                       architecture=varin['architecture'],
                                                                       obs_cal=obs_cal,
                                                                       decoding_method='viterbi')
        else:
            eval_results_decoding_path = eval_results_path + str(ii)

        precision_onset, recall_onset, F1_onset, \
        precision, recall, F1, \
            = eval_write_2_txt(eval_result_file_name=join(eval_results_decoding_path, 'results.csv'),
                               segSyllable_path=eval_results_decoding_path,
                               label=eval_label,
                               decoding_method='viterbi')

        list_precision_onset_25.append(precision_onset[0])
        list_precision_onset_5.append(precision_onset[1])
        list_recall_onset_25.append(recall_onset[0])
        list_recall_onset_5.append(recall_onset[1])
        list_F1_onset_25.append(F1_onset[0])
        list_F1_onset_5.append(F1_onset[1])
        list_precision_25.append(precision[0])
        list_precision_5.append(precision[1])
        list_recall_25.append(recall[0])
        list_recall_5.append(recall[1])
        list_F1_25.append(F1[0])
        list_F1_5.append(F1[1])

    return list_precision_onset_25, \
           list_recall_onset_25, \
           list_F1_onset_25, \
           list_precision_25, \
           list_recall_25, \
           list_F1_25, \
           list_precision_onset_5, \
           list_recall_onset_5, \
           list_F1_onset_5, \
           list_precision_5, \
           list_recall_5, \
           list_F1_5


def peak_picking_subroutine(test_nacta_2017, test_nacta, th, obs_cal):
    """Peak picking routine,
    five folds evaluation"""
    from src.utilFunctions import append_or_write
    import csv

    eval_result_file_name = join(jingju_results_path,
                                 varin['sample_weighting'],
                                 cnnModel_name + '_peakPicking_threshold_results.txt')

    list_recall_onset_25, list_precision_onset_25, list_F1_onset_25 = [], [], []
    list_recall_onset_5, list_precision_onset_5, list_F1_onset_5 = [], [], []
    list_recall_25, list_precision_25, list_F1_25 = [], [], []
    list_recall_5, list_precision_5, list_F1_5 = [], [], []

    for ii in range(5):

        if obs_cal == 'tocal':
            if 'schluter' in varin['architecture']:
                scaler = cPickle.load(gzip.open(full_path_mfccBands_2D_scaler_onset+str(ii)+'.pickle.gz'))

            model_keras_cnn_0 = load_model(full_path_keras_cnn_0 + str(ii) + '.h5')
        else:
            model_keras_cnn_0 = None
            scaler = None

        if varin['dataset'] != 'ismir':
            # nacta2017
            batch_process_onset_detection(wav_path=nacta2017_wav_path,
                                          textgrid_path=nacta2017_textgrid_path,
                                          score_path=nacta2017_score_pinyin_path,
                                          test_recordings=test_nacta_2017,
                                          model_keras_cnn_0=model_keras_cnn_0,
                                          cnnModel_name=cnnModel_name + str(ii),
                                          eval_results_path=eval_results_path + str(ii),
                                          scaler=scaler,
                                          architecture=varin['architecture'],
                                          threshold=th,
                                          obs_cal=obs_cal,
                                          decoding_method='peakPicking')

        eval_results_decoding_path = batch_process_onset_detection(wav_path=nacta_wav_path,
                                                                   textgrid_path=nacta_textgrid_path,
                                                                   score_path=nacta_score_pinyin_path,
                                                                   test_recordings=test_nacta,
                                                                   model_keras_cnn_0=model_keras_cnn_0,
                                                                   cnnModel_name=cnnModel_name + str(ii),
                                                                   eval_results_path=eval_results_path + str(ii),
                                                                   scaler=scaler,
                                                                   architecture=varin['architecture'],
                                                                   threshold=th,
                                                                   obs_cal=obs_cal,
                                                                   decoding_method='peakPicking')

        append_write = append_or_write(eval_result_file_name)
        with open(eval_result_file_name, append_write) as testfile:
            csv_writer = csv.writer(testfile)
            csv_writer.writerow([th])

        precision_onset, recall_onset, F1_onset, \
        precision, recall, F1, \
            = eval_write_2_txt(eval_result_file_name,
                               eval_results_decoding_path,
                               label=False,
                               decoding_method='peakPicking')

        list_precision_onset_25.append(precision_onset[0])
        list_precision_onset_5.append(precision_onset[1])
        list_recall_onset_25.append(recall_onset[0])
        list_recall_onset_5.append(recall_onset[1])
        list_F1_onset_25.append(F1_onset[0])
        list_F1_onset_5.append(F1_onset[1])
        list_precision_25.append(precision[0])
        list_precision_5.append(precision[1])
        list_recall_25.append(recall[0])
        list_recall_5.append(recall[1])
        list_F1_25.append(F1[0])
        list_F1_5.append(F1[1])

    return list_precision_onset_25, \
           list_recall_onset_25, \
           list_F1_onset_25, \
           list_precision_25, \
           list_recall_25, \
           list_F1_25, \
           list_precision_onset_5, \
           list_recall_onset_5, \
           list_F1_onset_5, \
           list_precision_5, \
           list_recall_5, \
           list_F1_5


def viterbi_label_eval(test_nacta_2017, test_nacta, eval_label, obs_cal):
    """evaluate viterbi onset detection"""

    list_precision_onset_25, list_recall_onset_25, list_F1_onset_25, list_precision_25, list_recall_25, list_F1_25, \
    list_precision_onset_5, list_recall_onset_5, list_F1_onset_5, list_precision_5, list_recall_5, list_F1_5 = \
        viterbi_subroutine(test_nacta_2017=test_nacta_2017,
                           test_nacta=test_nacta,
                           eval_label=eval_label,
                           obs_cal=obs_cal)

    postfix_statistic_sig = 'label' if eval_label else 'nolabel'

    pickle.dump(list_F1_onset_25,
                open(join('./statisticalSignificance/data/jingju', varin['sample_weighting'],
                          cnnModel_name + '_' + 'viterbi' + '_' + postfix_statistic_sig + '.pkl'), 'w'))

    write_results_2_txt_jingju(join(jingju_results_path,
                                    varin['sample_weighting'],
                                    cnnModel_name + '_viterbi' + '_' + postfix_statistic_sig + '.txt'),
                               postfix_statistic_sig,
                     'viterbi',
                               list_precision_onset_25,
                               list_recall_onset_25,
                               list_F1_onset_25,
                               list_precision_25,
                               list_recall_25,
                               list_F1_25,
                               list_precision_onset_5,
                               list_recall_onset_5,
                               list_F1_onset_5,
                               list_precision_5,
                               list_recall_5,
                               list_F1_5)


def peak_picking_eval(test_nacta_2017, test_nacta, obs_cal):
    """evaluate the peak picking results,
    search for the best threshold"""

    # Step1: coarse scan the best threshold, step 0.1
    best_F1_onset_25, best_th = 0, 0

    for th in range(1, 9):
        th *= 0.1

        _, _, list_F1_onset_25, _, _, _, _, _, _, _, _, _ = peak_picking_subroutine(test_nacta_2017=test_nacta_2017,
                                                                                    test_nacta=test_nacta,
                                                                                    th=th,
                                                                                    obs_cal=obs_cal)

        if np.mean(list_F1_onset_25) > best_F1_onset_25:
            best_th = th
            best_F1_onset_25 = np.mean(list_F1_onset_25)

    # Step 2: finer scan the best threshold
    for th in range(int((best_th - 0.1) * 100), int((best_th + 0.1) * 100)):
        th *= 0.01

        _, _, list_F1_onset_25, _, _, _, _, _, _, _, _, _ = peak_picking_subroutine(test_nacta_2017=test_nacta_2017,
                                                                                    test_nacta=test_nacta,
                                                                                    th=th,
                                                                                    obs_cal=obs_cal)

        if np.mean(list_F1_onset_25) > best_F1_onset_25:
            best_th = th
            best_F1_onset_25 = np.mean(list_F1_onset_25)

    # Step 3: get the statistics of the best th
    list_precision_onset_25, list_recall_onset_25, list_F1_onset_25, list_precision_25, list_recall_25, list_F1_25, \
    list_precision_onset_5, list_recall_onset_5, list_F1_onset_5, list_precision_5, list_recall_5, list_F1_5 = \
        peak_picking_subroutine(test_nacta_2017=test_nacta_2017,
                                test_nacta=test_nacta,
                                th=best_th,
                                obs_cal=obs_cal)

    print('best_th', best_th)

    # statistical significance data
    pickle.dump(list_F1_onset_25,
                open(join('./statisticalSignificance/data/jingju',
                          varin['sample_weighting'],
                          cnnModel_name + '_peakPickingMadmom.pkl'), 'w'))

    # save the results
    write_results_2_txt_jingju(join(jingju_results_path,
                                    varin['sample_weighting'],
                                    cnnModel_name + '_peakPickingMadmom' + '.txt'),
                               str(best_th),
                     'peakPicking',
                               list_precision_onset_25,
                               list_recall_onset_25,
                               list_F1_onset_25,
                               list_precision_25,
                               list_recall_25,
                               list_F1_25,
                               list_precision_onset_5,
                               list_recall_onset_5,
                               list_F1_onset_5,
                               list_precision_5,
                               list_recall_5,
                               list_F1_5)


if __name__ == '__main__':

    # load the test recordings
    test_nacta_2017, test_nacta = getTestRecordingsScoreDurCorrectionArtistAlbumFilter()

    if 'schluter' not in varin['architecture']:
        scaler = pickle.load(open(full_path_mfccBands_2D_scaler_onset, 'rb'))

    # calculate the ODF only in the first round
    # then we can load them for saving time
    obs_cal = 'tocal'

    # evaluate label
    viterbi_label_eval(test_nacta_2017=test_nacta_2017,
                       test_nacta=test_nacta,
                       eval_label=True,
                       obs_cal=obs_cal)

    obs_cal = 'toload'

    # do not evaluate label
    viterbi_label_eval(test_nacta_2017=test_nacta_2017,
                       test_nacta=test_nacta,
                       eval_label=False,
                       obs_cal=obs_cal)

    # peak picking evaluation
    peak_picking_eval(test_nacta_2017=test_nacta_2017,
                      test_nacta=test_nacta,
                      obs_cal=obs_cal)
