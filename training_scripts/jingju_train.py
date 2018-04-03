import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './jingju_crnn')))

from models import train_model_validation
from models import finetune_model_validation
from jingju_crnn import jingju_crnn_basecode


nlen = 15
input_dim = (80, nlen)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="To train the jingju models.")

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

    parser.add_argument("--path_pretrained",
                        type=str,
                        default=None,
                        help="Path of the pretrained model")

    args = parser.parse_args()

    filename_train_validation_set = os.path.join(args.path_input, 'feature_jingju.h5')
    filename_labels_train_validation_set = os.path.join(args.path_input, 'labels_jingju.pkl')
    filename_sample_weights = os.path.join(args.path_input, 'sample_weights_jingju.pkl')

    for running_time in range(5):

        file_path_model = os.path.join(args.path_output, args.architecture + str(running_time) + '.h5')
        file_path_log = os.path.join(args.path_output, args.architecture + str(running_time) + '.csv')

        if args.architecture == 'baseline':
            train_model_validation(filename_train_validation_set,
                                   filename_labels_train_validation_set,
                                   filename_sample_weights,
                                   filter_density=1,
                                   dropout=0.5,
                                   input_shape=input_dim,
                                   file_path_model=file_path_model,
                                   filename_log=file_path_log,
                                   model_name='jan_original',
                                   activation_dense='sigmoid',
                                   channel=1)
        elif args.architecture == 'relu_dense':
            train_model_validation(filename_train_validation_set,
                                   filename_labels_train_validation_set,
                                   filename_sample_weights,
                                   filter_density=1,
                                   dropout=0.5,
                                   input_shape=input_dim,
                                   file_path_model=file_path_model,
                                   filename_log=file_path_log,
                                   model_name='jan_original',
                                   activation_dense='relu',
                                   channel=1)
        elif args.architecture == 'no_dense':
            train_model_validation(filename_train_validation_set,
                                   filename_labels_train_validation_set,
                                   filename_sample_weights,
                                   filter_density=1,
                                   dropout=0.5,
                                   input_shape=input_dim,
                                   file_path_model=file_path_model,
                                   filename_log=file_path_log,
                                   model_name='jan_original',
                                   activation_dense='sigmoid',
                                   dense=False,
                                   channel=1)
        elif args.architecture == 'temporal':
            train_model_validation(filename_train_validation_set,
                                   filename_labels_train_validation_set,
                                   filename_sample_weights,
                                   filter_density=1,
                                   dropout=0.5,
                                   input_shape=input_dim,
                                   file_path_model=file_path_model,
                                   filename_log=file_path_log,
                                   model_name='jordi_temporal_schluter',
                                   activation_dense='sigmoid',
                                   channel=1)
        elif args.architecture == '5_layers_cnn':
            train_model_validation(filename_train_validation_set,
                                   filename_labels_train_validation_set,
                                   filename_sample_weights,
                                   filter_density=1,
                                   dropout=0.5,
                                   input_shape=input_dim,
                                   file_path_model=file_path_model,
                                   filename_log=file_path_log,
                                   model_name='jan_original',
                                   channel=1,
                                   deep='5_layers_cnn')
        elif args.architecture == '9_layers_cnn':
            train_model_validation(filename_train_validation_set,
                                   filename_labels_train_validation_set,
                                   filename_sample_weights,
                                   filter_density=1,
                                   dropout=0.5,
                                   input_shape=input_dim,
                                   file_path_model=file_path_model,
                                   filename_log=file_path_log,
                                   model_name='jan_original',
                                   channel=1,
                                   deep='9_layers_cnn')
        elif args.architecture == 'retrained':
            finetune_model_validation(filename_train_validation_set,
                                      filename_labels_train_validation_set,
                                      filename_sample_weights,
                                      filter_density=1,
                                      dropout=0.5,
                                      input_shape=input_dim,
                                      file_path_model=file_path_model,
                                      filename_log=file_path_log,
                                      model_name='retrained',
                                      path_model=args.path_pretrained,
                                      deep='5_layers_cnn',
                                      dense=True,
                                      channel=1)
        elif args.architecture == 'feature_extractor_a':
            finetune_model_validation(filename_train_validation_set,
                                      filename_labels_train_validation_set,
                                      filename_sample_weights,
                                      filter_density=1,
                                      dropout=0.5,
                                      input_shape=input_dim,
                                      file_path_model=file_path_model,
                                      filename_log=file_path_log,
                                      model_name='feature_extractor_a',
                                      path_model=args.path_pretrained,
                                      deep='5_layers_cnn',
                                      dense=True,
                                      channel=1)
        elif args.architecture == 'feature_extractor_b':
            finetune_model_validation(filename_train_validation_set,
                                      filename_labels_train_validation_set,
                                      filename_sample_weights,
                                      filter_density=1,
                                      dropout=0.5,
                                      input_shape=input_dim,
                                      file_path_model=file_path_model,
                                      filename_log=file_path_log,
                                      model_name='feature_extractor_a',
                                      path_model=args.path_pretrained,
                                      deep='5_layers_cnn',
                                      dense=True,
                                      channel=1)
        elif args.architecture == 'bidi_lstms_100':
            jingju_crnn_basecode.run_training_process(path_input=args.path_input,
                                                      path_output=args.path_output,
                                                      ii=running_time,
                                                      len_seq=100)
        elif args.architecture == 'bidi_lstms_200':
            jingju_crnn_basecode.run_training_process(path_input=args.path_input,
                                                      path_output=args.path_output,
                                                      ii=running_time,
                                                      len_seq=200)
        elif args.architecture == 'bidi_lstms_400':
            jingju_crnn_basecode.run_training_process(path_input=args.path_input,
                                                      path_output=args.path_output,
                                                      ii=running_time,
                                                      len_seq=400)
        else:
            pass
