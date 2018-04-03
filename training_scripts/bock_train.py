import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './jingju_crnn')))

from models import train_model_validation
from models import finetune_model_validation
from data_preparation_schluter import getTrainingFilenames
from data_preparation_schluter import concatenateFeatureLabelSampleweights
from data_preparation_schluter import saveFeatureLabelSampleweights
from bock_crnn import bock_crnn_basecode

nlen = 15
input_dim = (80, nlen)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="To train the bock models.")

    parser.add_argument("-a",
                        "--architecture",
                        type=str,
                        default='baseline',
                        choices=['baseline', 'relu_dense', 'no_dense', 'temporal', 'bidi_lstms_100',
                                 'bidi_lstms_200', 'bidi_lstms_400', '9_layers_cnn', '5_layers_cnn',
                                 'retrained', 'feature_extractor_a', 'feature_extractor_b'],
                        help="choose the architecture.")

    parser.add_argument("--path_input",
                        type=str,
                        help="Input path where you put the training data")

    parser.add_argument("--path_output",
                        type=str,
                        help="Output path where you store the models and logs")

    parser.add_argument("--path_cv",
                        type=str,
                        help="Cross validation split path")

    parser.add_argument("--path_annotation",
                        type=str,
                        help="Annotation path")

    parser.add_argument("--path_pretrained",
                        type=str,
                        default=None,
                        help="Path of the pretrained model")

    args = parser.parse_args()

    for ii_fold in range(0, 8):

        if args.architecture in ['baseline', 'relu_dense', 'no_dense', 'temporal', '9_layers_cnn', '5_layers_cnn',
                                 'retrained', 'feature_extractor_a', 'feature_extractor_b']:

            # organize dataset -----------------------------------------------------------------------------------------
            test_cv_filename = os.path.join(args.path_cv, '8-fold_cv_random_' + str(ii_fold) + '.fold')
            train_fns = getTrainingFilenames(args.path_annotation, test_cv_filename)
            feature_all, label_all, sample_weights_all, scaler = concatenateFeatureLabelSampleweights(train_fns,
                                                                                                      args.path_input,
                                                                                                      n_pattern=15,
                                                                                                      nlen=7,
                                                                                                      scaling=True)

            # create the temp bock folder if not exists
            temp_folder_bock = os.path.join(args.path_input, 'temp')
            if not os.path.exists(temp_folder_bock):
                os.makedirs(temp_folder_bock)

            filename_train_validation_set = os.path.join(temp_folder_bock, 'feature_bock_' + str(ii_fold) + '.h5')
            filename_labels_train_validation_set = os.path.join(temp_folder_bock, 'labels_bock_' + str(ii_fold) + '.pkl')
            filename_sample_weights = os.path.join(temp_folder_bock, 'sample_weights_bock_' + str(ii_fold) + '.pkl')
            filename_scaler = os.path.join(temp_folder_bock, 'scaler_bock_' + str(ii_fold) + '.pkl')

            saveFeatureLabelSampleweights(feature_all, label_all, sample_weights_all, scaler,
                                          filename_train_validation_set, filename_labels_train_validation_set,
                                          filename_sample_weights, filename_scaler)

            print('Finished organizing dataset.')

            file_path_model = os.path.join(args.path_output, args.architecture + str(ii_fold) + '.h5')
            file_path_log = os.path.join(args.path_output, args.architecture + str(ii_fold) + '.csv')

            # architecture -------------------------------------------------------------------------------------------------
            if args.architecture in ['baseline', 'relu_dense', 'no_dense', 'temporal', '9_layers_cnn', '5_layers_cnn']:

                train_model_validation(filename_train_validation_set,
                                       filename_labels_train_validation_set,
                                       filename_sample_weights,
                                       filter_density=1,
                                       dropout=0.5,
                                       input_shape=input_dim,
                                       file_path_model=file_path_model,
                                       filename_log=file_path_log,
                                       model_name=args.architecture,
                                       channel=1)

            elif args.architecture in ['retrained', 'feature_extractor_a', 'feature_extractor_b']:
                finetune_model_validation(filename_train_validation_set,
                                          filename_labels_train_validation_set,
                                          filename_sample_weights,
                                          filter_density=1,
                                          dropout=0.5,
                                          input_shape=input_dim,
                                          file_path_model=file_path_model,
                                          filename_log=file_path_log,
                                          model_name=args.architecture,
                                          path_model=args.path_pretrained,
                                          channel=1)

            os.remove(filename_train_validation_set)
            os.remove(filename_labels_train_validation_set)
            os.remove(filename_sample_weights)

        else:
            if args.architecture == 'bidi_lstms_100':
                bock_crnn_basecode.run_bock_training(path_input=args.path_input,
                                                     path_output=args.path_output,
                                                     bock_cv_path=args.path_cv,
                                                     bock_annotations_path=args.path_annotation,
                                                     len_seq=100,
                                                     ii=ii_fold)
            elif args.architecture == 'bidi_lstms_200':
                bock_crnn_basecode.run_bock_training(path_input=args.path_input,
                                                     path_output=args.path_output,
                                                     bock_cv_path=args.path_cv,
                                                     bock_annotations_path=args.path_annotation,
                                                     len_seq=200,
                                                     ii=ii_fold)
            elif args.architecture == 'bidi_lstms_400':
                bock_crnn_basecode.run_bock_training(path_input=args.path_input,
                                                     path_output=args.path_output,
                                                     bock_cv_path=args.path_cv,
                                                     bock_annotations_path=args.path_annotation,
                                                     len_seq=400,
                                                     ii=ii_fold)
            else:
                pass