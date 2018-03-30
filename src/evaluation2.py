# -*- coding: utf-8 -*-

'''
Syllable segmentation evaluation: landmark and boundary evaluations
Only evaluate boundary onset

[1] A new hybrid approach for automatic speech signal segmentation
using silence signal detection, energy convex hull, and spectral variation

[2] Syll-O-Matic: An adaptive time-frequency representation
for the automatic segmentation of speech into syllables

[3] EVALUATION FRAMEWORK FOR AUTOMATIC SINGING
TRANSCRIPTION
'''

import numpy as np

def landmarkEval(groundtruthBoundaries, landmarks):
    '''
    :param groundtruthBoundaries:   row[0] syllable start time in second,
                                    row[1] syllable end time in second,
                                    column syllables
    :param landmarks:               landmarks in second
    :return:                        number of landmarks,
                                    ground truth syllables,
                                    correct syllables,
                                    insertions,
                                    deletions
    '''

    numLandmarks            = len(landmarks)
    numGroundtruthSyllables = len(groundtruthBoundaries)

    correctlist             = [0]*numGroundtruthSyllables

    for lm in landmarks:
        for idx, gtb in enumerate(groundtruthBoundaries):
            if gtb[0] < lm < gtb[1]:
                correctlist[idx] = 1                                # found landmark for boundary idx

    numCorrect      = sum(correctlist)
    numInsertion    = numLandmarks - numCorrect
    numDeletion     = numGroundtruthSyllables - numCorrect

    return numLandmarks, numGroundtruthSyllables, numCorrect, numInsertion, numDeletion

def boundaryEval(groundtruthBoundaries, detectedBoundaries, tolerance, label):
    '''
    :param groundtruthBoundaries:   col[0] syllable start time in second,
                                    col[1] syllable end time in second,
                                    column syllables
    :param detectedBoundaries:      same structure as groundtruthBoundaries
    :param tolerance:               tolerance of evaluation
    :return:                        number of detected syllables,
                                    ground truth syllables,
                                    correct boundaries,
                                    insertions,
                                    deletions
    '''

    numDetectedBoundaries       = len(detectedBoundaries)
    numGroundtruthBoundaries    = len(groundtruthBoundaries)

    correctlist                 = [0]*numDetectedBoundaries
    onsetCorrectlist            = [0]*numDetectedBoundaries
    offsetCorrectlist           = [0]*numDetectedBoundaries

    for gtb in groundtruthBoundaries:
        for idx, db in enumerate(detectedBoundaries):
            onsetTh     = tolerance                                          # onset threshold
            offsetTh    = max(tolerance,(gtb[1]-gtb[0])*0.2)                 # offset threshold
            # if abs(db[0]-gtb[0])<onsetTh and abs(db[1]-gtb[1])<offsetTh and db[2] == gtb[2]:
            if abs(db[0]-gtb[0])<onsetTh and abs(db[1]-gtb[1])<offsetTh:
                if label:
                    if db[2] == gtb[2]:
                        correctlist[idx] = 1  # found landmark for boundary idx
                else:
                    correctlist[idx] = 1  # found landmark for boundary idx

            if abs(db[0]-gtb[0])<onsetTh:
                if label:
                    if db[2] == gtb[2]:
                        onsetCorrectlist[idx] = 1
                else:
                    onsetCorrectlist[idx] = 1
            if abs(db[1]-gtb[1])<offsetTh:
                offsetCorrectlist[idx] = 1

    numCorrect          = sum(correctlist)
    numOnsetCorrect     = sum(onsetCorrectlist)
    numOffsetCorrect    = sum(offsetCorrectlist)
    numInsertion        = numDetectedBoundaries - numCorrect
    numDeletion         = numGroundtruthBoundaries - numCorrect

    return numDetectedBoundaries, numGroundtruthBoundaries, numCorrect, numOnsetCorrect, numOffsetCorrect, numInsertion, numDeletion, correctlist

def onsetEval(groundtruthOnsets, detectedOnsets, tolerance):
    '''
    :param groundtruthBoundaries:   col syllable start time in second,
    :param detectedBoundaries:      same structure as groundtruthBoundaries
    :param tolerance:               tolerance of evaluation
    :return:                        number of detected syllables,
                                    ground truth syllables,
                                    correct boundaries,
                                    insertions,
                                    deletions
    '''

    numDetectedOnsets       = len(detectedOnsets)
    numGroundtruthOnsets    = len(groundtruthOnsets)

    correctlist                 = [0]*numDetectedOnsets

    for gto in groundtruthOnsets:
        for idx, do in enumerate(detectedOnsets):
            onsetTh     = tolerance                                          # onset threshold
            # if abs(db[0]-gtb[0])<onsetTh and abs(db[1]-gtb[1])<offsetTh and db[2] == gtb[2]:
            if abs(do-gto)<onsetTh:
                correctlist[idx] = 1  # found landmark for boundary idx

    numCorrect          = sum(correctlist)
    numInsertion        = numDetectedOnsets - numCorrect
    numDeletion         = numGroundtruthOnsets - numCorrect

    return numDetectedOnsets, numGroundtruthOnsets, numCorrect, numInsertion, numDeletion, correctlist

def metrics(numDetected, numGroundtruth, numCorrect):
    recall = (numCorrect/float(numGroundtruth))*100
    precision = (numCorrect/float(numDetected))*100
    if precision==0 and recall == 0:
        F1 = 0
    else:
        F1 = 2*(precision*recall)/(precision+recall)

    return recall,precision,F1