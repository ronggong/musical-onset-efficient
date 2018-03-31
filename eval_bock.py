from src.utilFunctions import getRecordings
from src.schluterParser import annotationCvParser
from src.evaluation2 import metrics
from mir_eval.onset import f_measure
from os.path import join
import numpy as np


def eval_bock(architecture,
              detection_results_path,
              bock_annotations_path):
    """
    evaluate bock model for each fold
    :param model_type:
    :return: recall, precision and f1 for each fold
    """

    sumNumDetectedOnsets = 0
    sumNumGroundtruthOnsets = 0
    sumNumCorrectOnsets = 0

    recall_precision_f1_folds = []
    jj = 0

    for ii in range(8):
        model_name = architecture + str(ii)
        detected_onset_path = getRecordings(join(detection_results_path, model_name))
        # groundtruth_onset_filenames = getRecordings(schluter_annotations_path)

        sumNumDetectedOnsets_fold = 0
        sumNumGroundtruthOnsets_fold = 0
        sumNumCorrectOnsets_fold = 0
        for dop in detected_onset_path:
            fn = dop.replace('.syll', '')
            # fn = dop.replace('.onsets', '')
            detected_fn = join(detection_results_path, model_name, dop + '.lab')
            groundtruth_fn = join(bock_annotations_path, fn + '.onsets')

            detected_onsets = annotationCvParser(detected_fn)
            groundtruth_onsets = annotationCvParser(groundtruth_fn)

            detected_onsets = [float(do) for do in detected_onsets]
            groundtruth_onsets = [float(go) for go in groundtruth_onsets]

            # numDetectedOnsets, numGroundtruthOnsets, numCorrect, numInsertion, numDeletion, correctlist = \
            #     onsetEval(groundtruth_onsets, detected_onsets, 0.025)


            if len(detected_onsets) == 0:
                matching = []
            else:
                _, _, _, matching = f_measure(np.array(groundtruth_onsets), np.array(detected_onsets), 0.025)
            numCorrect = len(matching)
            numGroundtruthOnsets = len(groundtruth_onsets)
            numDetectedOnsets = len(detected_onsets)

            sumNumDetectedOnsets += numDetectedOnsets
            sumNumGroundtruthOnsets += numGroundtruthOnsets
            sumNumCorrectOnsets += numCorrect

            sumNumDetectedOnsets_fold += numDetectedOnsets
            sumNumGroundtruthOnsets_fold += numGroundtruthOnsets
            sumNumCorrectOnsets_fold += numCorrect

            jj += 1

        recall_fold, precision_fold, F1_fold = metrics(sumNumDetectedOnsets_fold,
                                                        sumNumGroundtruthOnsets_fold,
                                                        sumNumCorrectOnsets_fold)
        recall_precision_f1_folds.append([recall_fold, precision_fold, F1_fold])

    recall_overall, precision_overall, F1_overall = metrics(sumNumDetectedOnsets,
                                                            sumNumGroundtruthOnsets,
                                                            sumNumCorrectOnsets)

    # print(model_type, "recal, precision, F1", recall_overall, precision_overall, F1_overall)
    # print(jj)
    return recall_precision_f1_folds, (recall_overall, precision_overall, F1_overall)


if __name__ == '__main__':
    # model_type = 'timbral'
    for model_type in ['timbral', 'temporal']:
        eval_bock(model_type)