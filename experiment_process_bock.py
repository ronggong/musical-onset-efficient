# -*- coding: utf-8 -*-
import cPickle
import gzip
import pickle
import os
import sys
import shutil
from os import makedirs
from os.path import exists
from os.path import join
from os.path import dirname

import numpy as np
from keras.models import load_model
from madmom.features.onsets import OnsetPeakPickingProcessor

from eval_bock import eval_bock
from plot_code import plot_schluter
from experiment_process_helper import write_results_2_txt_schluter
from experiment_process_helper import wav_annotation_loader_parser
from experiment_process_helper import peak_picking_detected_onset_saver_schluter
from experiment_process_helper import odf_calculation_schluter
from experiment_process_helper import odf_calculation_schluter_phrase

sys.path.append(os.path.join(os.path.dirname(__file__), "../src/"))

from parameters_schluter import *
from schluterParser import annotationCvParser
from utilFunctions import append_or_write

# from src.file_path_bock import *
# from src.file_path_shared import *


def batch_process_onset_detection(audio_path,
                                  annotation_path,
                                  filename,
                                  scaler_0,
                                  model_keras_cnn_0,
                                  model_name_0,
                                  model_name_1,
                                  architecture,
                                  detection_results_path,
                                  pp_threshold=0.54,
                                  channel=1,
                                  obs_cal='tocal'):
    """
    onset detection bock dataset
    :param audio_path: string, path where we store the audio
    :param annotation_path: string, path where we store annotation
    :param filename: string, audio filename
    :param scaler_0: sklearn object, StandardScaler
    :param model_keras_cnn_0: keras, .h5
    :param model_name_0: string
    :param model_name_1: string
    :param architecture: string, network architecture
    :param detection_results_path: string, where we store the detected results
    :param pp_threshold: float, peak picking threshold
    :param channel: int, 1 or 3, 3 is not used in the paper
    :param obs_cal: string, tocal or toload
    :return:
    """

    audio_filename, ground_truth_onset = wav_annotation_loader_parser(audio_path=audio_path,
                                                                      annotation_path=annotation_path,
                                                                      filename=filename,
                                                                      annotationCvParser=annotationCvParser)

    # create path to save ODF
    obs_path = join('./obs', model_name_0)
    obs_filename = filename + '.pkl'

    if obs_cal == 'tocal':

        obs_i, mfcc = odf_calculation_schluter(audio_filename=audio_filename,
                                               scaler_0=scaler_0,
                                               model_keras_cnn_0=model_keras_cnn_0,
                                               fs=fs,
                                               hopsize_t=hopsize_t,
                                               channel=channel,
                                               architecture=architecture)

        # save onset curve
        print('save onset curve ... ...')
        if not exists(obs_path):
            makedirs(obs_path)
        pickle.dump(obs_i, open(join(obs_path, obs_filename), 'w'))
    else:
        obs_i = pickle.load(open(join(obs_path, obs_filename), 'r'))

    obs_i = np.squeeze(obs_i)

    detected_onsets = peak_picking_detected_onset_saver_schluter(pp_threshold=pp_threshold,
                                                                 obs_i=obs_i,
                                                                 model_name_0=model_name_0,
                                                                 model_name_1=model_name_1,
                                                                 filename=filename,
                                                                 hopsize_t=hopsize_t,
                                                                 OnsetPeakPickingProcessor=OnsetPeakPickingProcessor,
                                                                 detection_results_path=detection_results_path)

    if varin['plot']:
        plot_schluter(mfcc=mfcc,
                      obs_i=obs_i,
                      hopsize_t=hopsize_t,
                      groundtruth_onset=ground_truth_onset,
                      detected_onsets=detected_onsets)


def batch_process_onset_detection_phrase(audio_path,
                                         annotation_path,
                                         filename,
                                         scaler_0,
                                         model_keras_cnn_0,
                                         model_name_0,
                                         model_name_1,
                                         stateful,
                                         len_seq,
                                         detection_results_path,
                                         pp_threshold=0.54,
                                         obs_cal='tocal'):
    """
    onset detection bock dataset in phrase level
    :param audio_path: string, path where we store the audio
    :param annotation_path: string, path where we store annotation
    :param filename: string, audio filename
    :param scaler_0: sklearn object, StandardScaler
    :param model_keras_cnn_0: keras, .h5
    :param model_name_0: string
    :param model_name_1: string
    :param stateful: where use stateful trained model, check stateful keras
    :param len_seq: int, input sequence length
    :param detection_results_path: string, where we store the detected results
    :param pp_threshold: float, peak picking threshold
    :param obs_cal: string, tocal or toload
    :return:
    """

    audio_filename, ground_truth_onset = wav_annotation_loader_parser(audio_path=audio_path,
                                                                      annotation_path=annotation_path,
                                                                      filename=filename,
                                                                      annotationCvParser=annotationCvParser)

    obs_path = join('./obs', model_name_0)
    obs_filename = filename + '.pkl'

    if obs_cal == 'tocal':

        obs_i, mfcc = odf_calculation_schluter_phrase(audio_filename=audio_filename,
                                                      scaler_0=scaler_0,
                                                      model_keras_cnn_0=model_keras_cnn_0,
                                                      fs=fs,
                                                      hopsize_t=hopsize_t,
                                                      len_seq=len_seq,
                                                      stateful=stateful)

        # save onset curve
        print('save onset curve ... ...')
        if not exists(obs_path):
            makedirs(obs_path)
        pickle.dump(obs_i, open(join(obs_path, obs_filename), 'w'))

    else:
        obs_i = pickle.load(open(join(obs_path, obs_filename), 'r'))

    obs_i = np.squeeze(obs_i)

    detected_onsets = peak_picking_detected_onset_saver_schluter(pp_threshold=pp_threshold,
                                                                 obs_i=obs_i,
                                                                 model_name_0=model_name_0,
                                                                 model_name_1=model_name_1,
                                                                 filename=filename,
                                                                 hopsize_t=hopsize_t,
                                                                 OnsetPeakPickingProcessor=OnsetPeakPickingProcessor,
                                                                 detection_results_path=detection_results_path)

    if varin['plot']:
        plot_schluter(mfcc=mfcc,
                      obs_i=obs_i,
                      hopsize_t=hopsize_t,
                      groundtruth_onset=ground_truth_onset,
                      detected_onsets=detected_onsets)


def schluter_eval_subroutine(nfolds,
                             pp_threshold,
                             obs_cal,
                             len_seq,
                             architecture,
                             bock_cv_path,
                             bock_cnn_model_path,
                             bock_audio_path,
                             bock_annotations_path,
                             bock_results_path,
                             detection_results_path,
                             jingju_cnn_model_path,
                             full_path_jingju_scaler):

    for ii in range(nfolds):
        # load scaler
        if 'bidi_lstms' not in architecture:  # not CRNN
            # only for jingju + schulter datasets trained model
            # scaler_name_0 = 'scaler_jan_madmom_simpleSampleWeighting_early_stopping_schluter_jingju_dataset_'
            # + str(ii)+'.pickle.gz'

            if 'pretrained' in architecture:
                scaler_0 = pickle.load(open(full_path_jingju_scaler, 'rb'))
            else:
                if 'temporal' in architecture:
                    scaler_name_0 = 'scaler_bock_' + str(ii) + '.pickle.gz'
                else:
                    scaler_name_0 = 'scaler_bock_temporal_' + str(ii) + '.pickle.gz'

                with gzip.open(join(bock_cnn_model_path, scaler_name_0), 'rb') as f:
                    scaler_0 = cPickle.load(f)
        else:  # CRNN
            scaler_name_0 = 'scaler_bock_phrase.pkl'
            scaler_0 = pickle.load(open(join(bock_cnn_model_path, scaler_name_0), 'rb'))

        # load model
        if 'pretrained' in architecture:
            model_name_0 = '5_layers_cnn0'
        else:
            model_name_0 = architecture + str(ii)

        model_name_1 = ''

        if obs_cal != 'tocal':
            model_keras_cnn_0 = None
            stateful = None
        else:
            if 'bidi_lstms' not in architecture:
                if 'pretrained' in architecture:
                    model_keras_cnn_0 = load_model(join(jingju_cnn_model_path, model_name_0 + '.h5'))
                else:
                    model_keras_cnn_0 = load_model(join(bock_cnn_model_path, model_name_0 + '.h5'))
                print(model_keras_cnn_0.summary())
            else:
                from training_scripts.models_CRNN import jan_original
                # initialize the model
                stateful = False
                bidi = True
                input_shape = (1, len_seq, 1, 80, 15)
                model_keras_cnn_0 = jan_original(filter_density=1,
                                                 dropout=0.5,
                                                 input_shape=input_shape,
                                                 batchNorm=False,
                                                 dense_activation='sigmoid',
                                                 channel=1,
                                                 stateful=stateful,
                                                 training=False,
                                                 bidi=bidi)
                # load weights
                model_keras_cnn_0.load_weights(join(bock_cnn_model_path, model_name_0 + '.h5'))

        # load cross validation filenames
        test_cv_filename = join(bock_cv_path, '8-fold_cv_random_' + str(ii) + '.fold')
        test_filenames = annotationCvParser(test_cv_filename)

        if 'pretrained' in architecture:
            model_name_0 = architecture + str(ii)

        # delete detection results path if it exists
        detection_results_path_model = join(detection_results_path, model_name_0)
        if os.path.exists(detection_results_path_model) and os.path.isdir(model_name_0):
            shutil.rmtree(model_name_0)

        for fn in test_filenames:
            if 'bidi_lstms' not in architecture:
                batch_process_onset_detection(audio_path=bock_audio_path,
                                              annotation_path=bock_annotations_path,
                                              filename=fn,
                                              scaler_0=scaler_0,
                                              model_keras_cnn_0=model_keras_cnn_0,
                                              model_name_0=model_name_0,
                                              model_name_1=model_name_1,
                                              pp_threshold=pp_threshold,
                                              channel=1,
                                              obs_cal=obs_cal,
                                              architecture=architecture,
                                              detection_results_path=detection_results_path)
            else:
                batch_process_onset_detection_phrase(audio_path=bock_audio_path,
                                                     annotation_path=bock_annotations_path,
                                                     filename=fn,
                                                     scaler_0=scaler_0,
                                                     model_keras_cnn_0=model_keras_cnn_0,
                                                     model_name_0=model_name_0,
                                                     model_name_1=model_name_1,
                                                     pp_threshold=pp_threshold,
                                                     stateful=stateful,
                                                     obs_cal=obs_cal,
                                                     len_seq=len_seq,
                                                     detection_results_path=detection_results_path)

    print('threshold', pp_threshold)
    recall_precision_f1_fold, recall_precision_f1_overall = eval_bock(architecture=architecture,
                                                                      detection_results_path=detection_results_path,
                                                                      bock_annotations_path=bock_annotations_path)

    log_path = join(bock_results_path,
                    varin['sample_weighting'],
                    architecture + '_' +
                    'threshold.txt')
    # log_path = join(schluter_results_path, weighting, 'schluter_jingju_model_threshold.txt')
    append_write = append_or_write(log_path)
    write_results_2_txt_schluter(log_path, append_write, pp_threshold, recall_precision_f1_overall)

    return recall_precision_f1_fold, recall_precision_f1_overall


def best_threshold_choosing(architecture,
                            len_seq,
                            bock_cv_path,
                            bock_cnn_model_path,
                            bock_audio_path,
                            bock_annotations_path,
                            bock_results_path,
                            detection_results_path,
                            jingju_cnn_model_path,
                            full_path_jingju_scaler):
    """recursively search for the best threshold"""
    best_F1, best_th = 0, 0

    # step 1: first calculate ODF and save
    pp_threshold = 0.1
    _, recall_precision_f1_overall \
        = schluter_eval_subroutine(nfolds=nfolds,
                                   pp_threshold=pp_threshold,
                                   obs_cal='tocal',
                                   len_seq=len_seq,
                                   architecture=architecture,
                                   bock_cv_path=bock_cv_path,
                                   bock_cnn_model_path=bock_cnn_model_path,
                                   bock_audio_path=bock_audio_path,
                                   bock_annotations_path=bock_annotations_path,
                                   bock_results_path=bock_results_path,
                                   detection_results_path=detection_results_path,
                                   jingju_cnn_model_path=jingju_cnn_model_path,
                                   full_path_jingju_scaler=full_path_jingju_scaler)

    if recall_precision_f1_overall[2] > best_F1:
        best_F1 = recall_precision_f1_overall[2]
        best_th = pp_threshold

    # step 2: load ODF and search
    for pp_threshold in range(2, 10):

        pp_threshold *= 0.1
        _, recall_precision_f1_overall \
            = schluter_eval_subroutine(nfolds=nfolds,
                                       pp_threshold=pp_threshold,
                                       obs_cal='toload',
                                       len_seq=len_seq,
                                       architecture=architecture,
                                       bock_cv_path=bock_cv_path,
                                       bock_cnn_model_path=bock_cnn_model_path,
                                       bock_audio_path=bock_audio_path,
                                       bock_annotations_path=bock_annotations_path,
                                       bock_results_path=bock_results_path,
                                       detection_results_path=detection_results_path,
                                       jingju_cnn_model_path=jingju_cnn_model_path,
                                       full_path_jingju_scaler=full_path_jingju_scaler)

        if recall_precision_f1_overall[2] > best_F1:
            best_F1 = recall_precision_f1_overall[2]
            best_th = pp_threshold

    # step 3: finer search the threshold
    best_recall_precision_f1_fold = None
    best_recall_precision_f1_overall = [0, 0, 0]
    for pp_threshold in range(int((best_th - 0.1) * 100), int((best_th + 0.1) * 100)):

        pp_threshold *= 0.01
        recall_precision_f1_fold, recall_precision_f1_overall \
            = schluter_eval_subroutine(nfolds=nfolds,
                                       pp_threshold=pp_threshold,
                                       obs_cal='toload',
                                       len_seq=len_seq,
                                       architecture=architecture,
                                       bock_cv_path=bock_cv_path,
                                       bock_cnn_model_path=bock_cnn_model_path,
                                       bock_audio_path=bock_audio_path,
                                       bock_annotations_path=bock_annotations_path,
                                       bock_results_path=bock_results_path,
                                       detection_results_path=detection_results_path,
                                       jingju_cnn_model_path=jingju_cnn_model_path,
                                       full_path_jingju_scaler=full_path_jingju_scaler)

        if recall_precision_f1_overall[2] > best_recall_precision_f1_overall[2]:
            best_recall_precision_f1_overall = recall_precision_f1_overall
            best_recall_precision_f1_fold = recall_precision_f1_fold
            best_th = pp_threshold

    return best_th, best_recall_precision_f1_fold, best_recall_precision_f1_overall


def results_saving(best_th,
                   best_recall_precision_f1_fold,
                   best_recall_precision_f1_overall,
                   architecture,
                   bock_results_path):
    # write recall precision f1 overall results

    # txt_filename_results_schluter = 'schluter_jingju_model.txt'

    # dump the evaluation results
    write_results_2_txt_schluter(join(bock_results_path,
                                      varin['sample_weighting'],
                                      architecture+'.txt'),
                                 'w',
                                 best_th,
                                 best_recall_precision_f1_overall)

    # filename_statistical_significance = 'schluter_jingju_model.pkl'

    # dump the statistical significance results
    pickle.dump(best_recall_precision_f1_fold,
                open(join('./statisticalSignificance/data',
                          'bock',
                          varin['sample_weighting'],
                          architecture+'.pkl'), 'w'))


def run_process_bock(architecture):

    len_seq = None

    if architecture == 'bidi_lstms_100':
        len_seq = 100  # sub-sequence length
    elif architecture == 'bidi_lstms_200':
        len_seq = 200
    elif architecture == 'bidi_lstms_400':
        len_seq = 400
    elif architecture not in ['baseline', 'no_dense', 'relu_dense', 'temporal', '9_layers_cnn', '5_layers_cnn',
                              'pretrained', 'retrained', 'feature_extractor_a', 'feature_extractor_b']:
        raise ValueError('There is no such architecture %s.' % architecture)

    root_path = join(dirname(__file__))

    # bock_dataset_root_path = '/Users/gong/Documents/MTG document/dataset/onsets'

    # bock_dataset_root_path = '/datasets/MTG/projects/compmusic/jingju_datasets/bock/'

    bock_dataset_root_path = '/media/gong/ec990efa-9ee0-4693-984b-29372dcea0d1/Data/RongGong/onsets'

    bock_audio_path = join(bock_dataset_root_path, 'audio')

    bock_cv_path = join(bock_dataset_root_path, 'splits')

    bock_annotations_path = join(bock_dataset_root_path, 'annotations')

    bock_cnn_model_path = join(root_path, 'pretrained_models', 'bock', varin['sample_weighting'])

    detection_results_path = join(root_path, 'eval', 'results')

    bock_results_path = join(root_path, 'eval', 'bock', 'results')

    # jingju model
    jingju_cnn_model_path = join(root_path, 'pretrained_models', 'jingju', varin['sample_weighting'])

    full_path_jingju_scaler = join(jingju_cnn_model_path, 'scaler_jan_no_rnn.pkl')

    best_th, best_recall_precision_f1_fold, best_recall_precision_f1_overall = \
        best_threshold_choosing(architecture=architecture,
                                len_seq=len_seq,
                                bock_cv_path=bock_cv_path,
                                bock_cnn_model_path=bock_cnn_model_path,
                                bock_audio_path=bock_audio_path,
                                bock_annotations_path=bock_annotations_path,
                                bock_results_path=bock_results_path,
                                detection_results_path=detection_results_path,
                                jingju_cnn_model_path=jingju_cnn_model_path,
                                full_path_jingju_scaler=full_path_jingju_scaler)
    results_saving(best_th=best_th,
                   best_recall_precision_f1_fold=best_recall_precision_f1_fold,
                   best_recall_precision_f1_overall=best_recall_precision_f1_overall,
                   architecture=architecture,
                   bock_results_path=bock_results_path)

