# -*- coding: utf-8 -*-
import sys
import os
import csv
import numpy as np

sys.path.append(os.path.realpath('./src/'))

import textgridParser
import labParser
import scoreParser
import evaluation2
from file_path_jingju_shared import *
from src.trainTestSeparation import getTestRecordingsScoreDurCorrectionArtistAlbumFilter
from src.trainTestSeparation import getTestTrainRecordingsArtistAlbumFilter


def batch_eval(annotation_path,
               segSyllable_path,
               score_path,
               groundtruth_path,
               eval_details_path,
               recordings,
               tolerance,
               label=False,
               decoding_method='viterbi'):

    sumDetectedBoundaries, sumGroundtruthPhrases, sumGroundtruthBoundaries, sumCorrect, sumOnsetCorrect, \
    sumOffsetCorrect, sumInsertion, sumDeletion = 0 ,0 ,0 ,0 ,0 ,0, 0, 0

    for artist_path, recording_name in recordings:

        if annotation_path:
            groundtruth_textgrid_file = os.path.join(annotation_path, artist_path, recording_name+'.TextGrid')
            groundtruth_lab_file_head = os.path.join(groundtruth_path, artist_path)
        else:
            groundtruth_syllable_lab = os.path.join(groundtruth_path, artist_path, recording_name+'.lab')

        detected_lab_file_head = os.path.join(segSyllable_path, artist_path,recording_name)

        score_file = os.path.join(score_path, artist_path,  recording_name+'.csv')

        # parse score
        if annotation_path:
            _, _, utterance_durations, bpm = scoreParser.csvScorePinyinParser(score_file)
        else:
            _, utterance_durations, bpm = scoreParser.csvDurationScoreParser(score_file)

        if eval_details_path:
            eval_result_details_file_head = os.path.join(eval_details_path, artist_path)

        if not os.path.isfile(score_file):
            print 'Score not found: ' + score_file
            continue

        if annotation_path:
            # create ground truth lab path, if not exist
            if not os.path.isdir(groundtruth_lab_file_head):
                os.makedirs(groundtruth_lab_file_head)

            if not os.path.isdir(eval_result_details_file_head):
                os.makedirs(eval_result_details_file_head)

            lineList = textgridParser.textGrid2WordList(groundtruth_textgrid_file, whichTier='line')
            utteranceList = textgridParser.textGrid2WordList(groundtruth_textgrid_file, whichTier='dianSilence')

            # parse lines of groundtruth
            nestedUtteranceLists, numLines, numUtterances = textgridParser.wordListsParseByLines(lineList, utteranceList)

            # create the ground truth lab files
            for idx, list in enumerate(nestedUtteranceLists):
                try:
                    print(bpm[idx])
                except IndexError:
                    continue

                if float(bpm[idx]):
                    print 'Creating ground truth lab ... ' + recording_name + ' phrase ' + str(idx+1)

                    ul = list[1]
                    firstStartTime = ul[0][0]
                    groundtruthBoundaries = [(np.array(ul_element[:2]) - firstStartTime).tolist() + [ul_element[2]] for ul_element in ul]
                    groundtruth_syllable_lab = join(groundtruth_lab_file_head, recording_name+'_'+str(idx+1)+'.syll.lab')

                    with open(groundtruth_syllable_lab, "wb") as text_file:
                        for gtbs in groundtruthBoundaries:
                            text_file.write("{0} {1} {2}\n".format(gtbs[0],gtbs[1],gtbs[2]))
        else:
            nestedUtteranceLists = [labParser.lab2WordList(groundtruth_syllable_lab, label=label)]


        for idx, list in enumerate(nestedUtteranceLists):
            try:
                print(bpm[idx])
            except IndexError:
                continue

            if float(bpm[idx]):
                print 'Evaluating... ' + recording_name + ' phrase ' + str(idx+1)

                if annotation_path:
                    ul = list[1]
                    firstStartTime  = ul[0][0]
                    groundtruthBoundaries = [(np.array(ul_element[:2]) - firstStartTime).tolist() + [ul_element[2]] for ul_element in ul]
                else:
                    firstStartTime = list[0][0]
                    groundtruthBoundaries = [(np.array(ul_element[:2]) - firstStartTime).tolist() + [ul_element[2]]for ul_element in list]

                detected_syllable_lab   = detected_lab_file_head+'_'+str(idx+1)+'.syll.lab'
                if not os.path.isfile(detected_syllable_lab):
                    print 'Syll lab file not found: ' + detected_syllable_lab
                    continue

                # read boundary detected lab into python list
                lab_label = True if decoding_method == 'viterbi' else False
                detectedBoundaries = labParser.lab2WordList(detected_syllable_lab, label=lab_label)

                numDetectedBoundaries, numGroundtruthBoundaries, numCorrect, numOnsetCorrect, numOffsetCorrect, \
                numInsertion, numDeletion, correct_list = evaluation2.boundaryEval(groundtruthBoundaries, detectedBoundaries, tolerance, label)

                sumDetectedBoundaries += numDetectedBoundaries
                sumGroundtruthBoundaries += numGroundtruthBoundaries
                sumGroundtruthPhrases += 1
                sumCorrect += numCorrect
                sumOnsetCorrect += numOnsetCorrect
                sumOffsetCorrect += numOffsetCorrect
                sumInsertion += numInsertion
                sumDeletion += numDeletion

                # if numCorrect/float(numGroundtruthBoundaries) < 0.7:
                print "Detected: {0}, Ground truth: {1}, Correct: {2}, Onset correct: {3}, " \
                      "Offset correct: {4}, Insertion: {5}, Deletion: {6}\n".\
                    format(numDetectedBoundaries, numGroundtruthBoundaries,numCorrect, numOnsetCorrect,
                           numOffsetCorrect, numInsertion, numDeletion)

    return sumDetectedBoundaries, sumGroundtruthBoundaries, sumGroundtruthPhrases, sumCorrect, sumOnsetCorrect, \
           sumOffsetCorrect, sumInsertion, sumDeletion


def stat_Add(sumDetectedBoundaries,
             sumGroundtruthBoundaries,
             sumGroundtruthPhrases,
             sumCorrect,
             sumOnsetCorrect,
             sumOffsetCorrect,
             sumInsertion,
             sumDeletion,
             DB,
             GB,
             GP,
             C,
             OnC,
             OffC,
             I,
             D):
    return sumDetectedBoundaries+DB, \
           sumGroundtruthBoundaries+GB, \
           sumGroundtruthPhrases+GP, \
           sumCorrect+C, \
           sumOnsetCorrect+OnC, \
           sumOffsetCorrect+OffC, \
           sumInsertion+I,\
           sumDeletion+D


def evaluation_test_dataset(segSyllablePath, tolerance, label, decoding_method):

    sumDetectedBoundaries, sumGroundtruthBoundaries, sumGroundtruthPhrases, sumCorrect, sumOnsetCorrect, sumOffsetCorrect, \
    sumInsertion, sumDeletion = 0, 0, 0, 0, 0, 0, 0, 0

    testNacta2017, testNacta, trainNacta2017, trainNacta = getTestTrainRecordingsArtistAlbumFilter()
    # testNacta2017, testNacta = getTestRecordingsScoreDurCorrectionArtistAlbumFilter()

    DB, GB, GP, C, OnC, OffC, I, D = batch_eval(annotation_path=nacta2017_textgrid_path,
                                                segSyllable_path=segSyllablePath,
                                                score_path=nacta2017_score_pinyin_path,
                                                groundtruth_path=nacta2017_groundtruthlab_path,
                                                eval_details_path=nacta2017_groundtruthlab_path,
                                                recordings=testNacta2017,
                                                tolerance=tolerance,
                                                label=label,
                                                decoding_method=decoding_method)

    sumDetectedBoundaries, sumGroundtruthBoundaries, \
    sumGroundtruthPhrases, sumCorrect, \
    sumOnsetCorrect, sumOffsetCorrect, \
    sumInsertion, sumDeletion = stat_Add(sumDetectedBoundaries,
                                         sumGroundtruthBoundaries,
                                         sumGroundtruthPhrases,
                                         sumCorrect,
                                         sumOnsetCorrect,
                                         sumOffsetCorrect,
                                         sumInsertion,
                                         sumDeletion,
                                         DB,
                                         GB,
                                         GP,
                                         C,
                                         OnC,
                                         OffC,
                                         I,
                                         D)

    DB, GB, GP, C, OnC, OffC, I, D = batch_eval(annotation_path=nacta_textgrid_path,
                                                segSyllable_path=segSyllablePath,
                                                score_path=nacta_score_pinyin_path,
                                                groundtruth_path=nacta_groundtruthlab_path,
                                                eval_details_path=nacta_eval_details_path,
                                                recordings=testNacta,
                                                tolerance=tolerance,
                                                label=label,
                                                decoding_method=decoding_method)

    sumDetectedBoundaries, sumGroundtruthBoundaries, \
    sumGroundtruthPhrases, sumCorrect, \
    sumOnsetCorrect, sumOffsetCorrect, \
    sumInsertion, sumDeletion = stat_Add(sumDetectedBoundaries,
                                         sumGroundtruthBoundaries,
                                         sumGroundtruthPhrases,
                                         sumCorrect,
                                         sumOnsetCorrect,
                                         sumOffsetCorrect,
                                         sumInsertion,
                                         sumDeletion,
                                         DB,
                                         GB,
                                         GP,
                                         C,
                                         OnC,
                                         OffC,
                                         I,
                                         D)

    print("Detected: {0}, Ground truth: {1}, Ground truth phrases: {2} Correct rate: {3}, Insertion rate: {4}, Deletion rate: {5}\n".
          format(sumDetectedBoundaries,
                 sumGroundtruthBoundaries,
                 sumGroundtruthPhrases,
                 sumCorrect / float(sumGroundtruthBoundaries),
                 sumInsertion / float(sumGroundtruthBoundaries),
                 sumDeletion / float(sumGroundtruthBoundaries)))

    return sumDetectedBoundaries, \
           sumGroundtruthBoundaries, \
           sumGroundtruthPhrases, \
           sumCorrect, \
           sumOnsetCorrect, \
           sumInsertion, \
           sumDeletion


def eval_write_2_txt(eval_result_file_name, segSyllable_path, label=True, decoding_method='viterbi'):
    from src.utilFunctions import append_or_write
    append_write = append_or_write(eval_result_file_name)
    tols = [0.025, 0.05]
    list_recall_onset, list_precision_onset, list_F1_onset = [], [], []
    list_recall, list_precision, list_F1 = [], [], []

    with open(eval_result_file_name, append_write) as testfile:
        csv_writer = csv.writer(testfile)
        for t in tols:
            detected, ground_truth, ground_truth_phrases, correct, onsetCorrect, insertion, deletion = \
                evaluation_test_dataset(segSyllable_path,
                                        tolerance=t,
                                        label=label,
                                        decoding_method=decoding_method)

            recall_onset,precision_onset,F1_onset = evaluation2.metrics(detected,ground_truth,onsetCorrect)
            recall,precision,F1 = evaluation2.metrics(detected,ground_truth,correct)

            csv_writer.writerow([t,detected, ground_truth, ground_truth_phrases, recall_onset,precision_onset,F1_onset])

            csv_writer.writerow([t,detected, ground_truth, ground_truth_phrases, recall,precision,F1])

            list_recall_onset.append(recall_onset)
            list_precision_onset.append(precision_onset)
            list_F1_onset.append(F1_onset)
            list_recall.append(recall)
            list_precision.append(precision)
            list_F1.append(F1)

    return list_precision_onset, list_recall_onset, list_F1_onset, \
            list_precision, list_recall, list_F1

