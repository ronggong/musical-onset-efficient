# -*- coding: utf-8 -*-
import cPickle
import gzip
import pickle
from os import makedirs
from os.path import exists

import numpy as np
from keras.models import load_model
from madmom.features.onsets import OnsetPeakPickingProcessor

from eval_schluter import eval_schluter
from src.file_path_schulter import *
from src.parameters_schluter import *

from src.schluterParser import annotationCvParser
from src.utilFunctions import append_or_write

from plot_code import plot_schluter
from experiment_process_helper import write_results_2_txt_schluter
from experiment_process_helper import wav_annotation_loader_parser
from experiment_process_helper import peak_picking_detected_onset_saver_schluter
from experiment_process_helper import odf_calculation_schluter
from experiment_process_helper import odf_calculation_schluter_phrase


def batch_process_onset_detection(audio_path,
                                  annotation_path,
                                  filename,
                                  scaler_0,
                                  model_keras_cnn_0,
                                  model_name_0,
                                  model_name_1,
                                  pp_threshold=0.54,
                                  channel=1,
                                  obs_cal='tocal'):
    """
    onset detection schluter dataset
    :param audio_path: string, path where we store the audio
    :param annotation_path: string, path where we store annotation
    :param filename: string, audio filename
    :param scaler_0: sklearn object, StandardScaler
    :param model_keras_cnn_0: keras, .h5
    :param model_name_0: string
    :param model_name_1: string
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
                                               no_dense_str=no_dense_str)

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
                                                                 eval_results_path=eval_results_path)

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
                                         pp_threshold=0.54,
                                         obs_cal='tocal'):
    """
    onset detection schluter dataset in phrase level
    :param audio_path: string, path where we store the audio
    :param annotation_path: string, path where we store annotation
    :param filename: string, audio filename
    :param scaler_0: sklearn object, StandardScaler
    :param model_keras_cnn_0: keras, .h5
    :param model_name_0: string
    :param model_name_1: string
    :param stateful: where use stateful trained model, check stateful keras
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
                                                                 eval_results_path=eval_results_path)

    if varin['plot']:
        plot_schluter(mfcc=mfcc,
                      obs_i=obs_i,
                      hopsize_t=hopsize_t,
                      groundtruth_onset=ground_truth_onset,
                      detected_onsets=detected_onsets)


def schluter_eval_subroutine(nfolds,
                             filter_shape_0,
                             weighting_str,
                             pp_threshold,
                             obs_cal):

    for ii in range(nfolds):
        if not phrase_eval:  # not CRNN
            model_name_str = 'schulter_' + \
                             filter_shape_0 + \
                             '_madmom_' + \
                             weighting_str + \
                             '_early_stopping_adam_cv' + \
                             relu_str + \
                             no_dense_str + \
                             deep_str + '_'

            # only for jingju + schulter datasets trained model
            # scaler_name_0 = 'scaler_jan_madmom_simpleSampleWeighting_early_stopping_schluter_jingju_dataset_'
            # + str(ii)+'.pickle.gz'

            scaler_name_0 = 'scaler_' + \
                            filter_shape_0 + \
                            '_madmom_' + \
                            weighting_str + \
                            '_early_stopping_' + \
                            str(ii) + '.pickle.gz'

        else:  # CRNN
            model_name_str = 'schulter_' + \
                             filter_shape_0 + \
                             '_madmom_' + \
                             weighting_str + \
                             '_early_stopping_adam_cv_phrase' + \
                             overlap_str + \
                             bidi_str

            scaler_name_0 = 'scaler_syllable_mfccBands2D_schluter_madmom_phrase.pkl'

        if 'jingju' in varin['architecture']:
            model_name_0 = 'keras.cnn_syllableSeg_jan_artist_filter_less_deep0'
        else:
            model_name_0 = model_name_str + str(ii)

        print(model_name_0)

        model_name_1 = ''

        test_cv_filename = join(schluter_cv_path, '8-fold_cv_random_' + str(ii) + '.fold')
        test_filenames = annotationCvParser(test_cv_filename)

        if obs_cal != 'tocal':
            model_keras_cnn_0 = None
            stateful = None
        else:
            if not phrase_eval:
                if 'jingju' in varin['architecture']:
                    model_keras_cnn_0 = load_model(join(jingju_cnn_model_path, model_name_0 + '.h5'))
                else:
                    model_keras_cnn_0 = load_model(join(schluter_cnn_model_path, model_name_0 + '.h5'))
                print(model_keras_cnn_0.summary())
            else:
                from training_scripts.models_CRNN import jan_original
                # initialize the model
                stateful = False if overlap else True
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
                model_keras_cnn_0.load_weights(join(schluter_cnn_model_path, model_name_0 + '.h5'))

        if not phrase_eval:
            if 'jingju' in varin['architecture']:
                scaler_0 = pickle.load(open(full_path_mfccBands_2D_scaler_onset, 'rb'))
            else:
                with gzip.open(join(schluter_cnn_model_path, scaler_name_0), 'rb') as f:
                    scaler_0 = cPickle.load(f)
        else:
            scaler_0 = pickle.load(open(join(schluter_cnn_model_path, scaler_name_0), 'rb'))

        if 'jingju' in varin['architecture']:
            model_name_str = 'schluter_' + model_name_0 + '_'
            model_name_0 = model_name_str + str(ii)

        for fn in test_filenames:
            if not phrase_eval:
                batch_process_onset_detection(audio_path=schluter_audio_path,
                                              annotation_path=schluter_annotations_path,
                                              filename=fn,
                                              scaler_0=scaler_0,
                                              model_keras_cnn_0=model_keras_cnn_0,
                                              model_name_0=model_name_0,
                                              model_name_1=model_name_1,
                                              pp_threshold=pp_threshold,
                                              channel=1,
                                              obs_cal=obs_cal)
            else:
                batch_process_onset_detection_phrase(audio_path=schluter_audio_path,
                                                     annotation_path=schluter_annotations_path,
                                                     filename=fn,
                                                     scaler_0=scaler_0,
                                                     model_keras_cnn_0=model_keras_cnn_0,
                                                     model_name_0=model_name_0,
                                                     model_name_1=model_name_1,
                                                     pp_threshold=pp_threshold,
                                                     stateful=stateful,
                                                     obs_cal=obs_cal)

    print('threshold', pp_threshold)
    recall_precision_f1_fold, recall_precision_f1_overall = eval_schluter(model_name_str)

    log_path = join(schluter_results_path,
                    varin['sample_weighting'],
                    'schluter' + '_' +
                    filter_shape_0 +
                    phrase_str +
                    overlap_str +
                    bidi_str +
                    relu_str +
                    deep_str +
                    no_dense_str + '_' +
                    'threshold.txt')
    # log_path = join(schluter_results_path, weighting, 'schluter_jingju_model_threshold.txt')
    append_write = append_or_write(log_path)
    write_results_2_txt_schluter(log_path, append_write, pp_threshold, recall_precision_f1_overall)

    return recall_precision_f1_fold, recall_precision_f1_overall


def best_threshold_choosing():
    """recursively search for the best threshold"""
    best_F1, best_th = 0, 0

    # step 1: first calculate ODF and save
    pp_threshold = 0.1
    _, recall_precision_f1_overall \
        = schluter_eval_subroutine(nfolds, filter_shape_0, weighting_str, pp_threshold, obs_cal='tocal')

    if recall_precision_f1_overall[2] > best_F1:
        best_F1 = recall_precision_f1_overall[2]
        best_th = pp_threshold

    # step 2: load ODF and search
    for pp_threshold in range(2, 10):

        pp_threshold *= 0.1
        _, recall_precision_f1_overall \
            = schluter_eval_subroutine(nfolds, filter_shape_0, weighting_str, pp_threshold, obs_cal='toload')

        if recall_precision_f1_overall[2] > best_F1:
            best_F1 = recall_precision_f1_overall[2]
            best_th = pp_threshold

    # step 3: finer search the threshold
    best_recall_precision_f1_fold = None
    best_recall_precision_f1_overall = [0, 0, 0]
    for pp_threshold in range(int((best_th - 0.1) * 100), int((best_th + 0.1) * 100)):

        pp_threshold *= 0.01
        recall_precision_f1_fold, recall_precision_f1_overall \
            = schluter_eval_subroutine(nfolds, filter_shape_0, weighting_str, pp_threshold, obs_cal='toload')

        if recall_precision_f1_overall[2] > best_recall_precision_f1_overall[2]:
            best_recall_precision_f1_overall = recall_precision_f1_overall
            best_recall_precision_f1_fold = recall_precision_f1_fold
            best_th = pp_threshold

    return best_th, best_recall_precision_f1_fold, best_recall_precision_f1_overall


def results_saving(best_th,
                   best_recall_precision_f1_fold,
                   best_recall_precision_f1_overall):
    # write recall precision f1 overall results
    txt_filename_results_schluter = 'schluter' + '_' + \
                                    filter_shape_0 + \
                                    phrase_str + \
                                    overlap_str + \
                                    bidi_str + \
                                    relu_str + \
                                    deep_str + \
                                    no_dense_str + '.txt'
    # txt_filename_results_schluter = 'schluter_jingju_model.txt'

    # dump the evaluation results
    write_results_2_txt_schluter(join(schluter_results_path,
                                      varin['sample_weighting'],
                                      txt_filename_results_schluter),
                                 'w',
                                 best_th,
                                 best_recall_precision_f1_overall)

    filename_statistical_significance = 'schluter' + '_' + \
                                        filter_shape_0 + \
                                        phrase_str + \
                                        overlap_str + \
                                        bidi_str + \
                                        relu_str + \
                                        deep_str + \
                                        no_dense_str + '.pkl'
    # filename_statistical_significance = 'schluter_jingju_model.pkl'

    # dump the statistical significance results
    pickle.dump(best_recall_precision_f1_fold,
                open(join('./statisticalSignificance/data',
                          'schluter',
                          varin['sample_weighting'],
                          filename_statistical_significance), 'w'))


if __name__ == '__main__':
    best_th, best_recall_precision_f1_fold, best_recall_precision_f1_overall = best_threshold_choosing()
    results_saving(best_th=best_th,
                   best_recall_precision_f1_fold=best_recall_precision_f1_fold,
                   best_recall_precision_f1_overall=best_recall_precision_f1_overall)
