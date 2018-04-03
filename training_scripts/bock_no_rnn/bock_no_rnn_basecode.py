import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preparation_schluter import getTrainingFilenames
from data_preparation_schluter import concatenateFeatureLabelSampleweights
from data_preparation_schluter import saveFeatureLabelSampleweights
from models import train_model_validation

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))


def run_training_process(model_name,
                         bock_cv_path,
                         bock_annotations_path,
                         bock_feature_path,
                         output_path,
                         ii):

    test_cv_filename = os.path.join(bock_cv_path, '8-fold_cv_random_' + str(ii) + '.fold')
    train_fns = getTrainingFilenames(bock_annotations_path, test_cv_filename)
    feature_all, label_all, sample_weights_all, scaler = concatenateFeatureLabelSampleweights(train_fns,
                                                                                              bock_feature_path,
                                                                                              n_pattern=15,
                                                                                              nlen=7,
                                                                                              scaling=True)

    # create the temp bock folder if not exists
    temp_folder_bock = os.path.join(bock_feature_path, 'temp')
    if not os.path.exists(temp_folder_bock):
        os.makedirs(temp_folder_bock)

    filename_train_validation_set = os.path.join(temp_folder_bock, 'feature_bock_' + str(ii) + '.h5')
    filename_labels_train_validation_set = os.path.join(temp_folder_bock, 'labels_bock_' + str(ii) + '.pkl')
    filename_sample_weights = os.path.join(temp_folder_bock, 'sample_weights_bock_' + str(ii) + '.pkl')
    filename_scaler = os.path.join(temp_folder_bock, 'scaler_bock_' + str(ii) + '.pkl')

    saveFeatureLabelSampleweights(feature_all, label_all, sample_weights_all, scaler,
                                  filename_train_validation_set, filename_labels_train_validation_set,
                                  filename_sample_weights, filename_scaler)

    print('Finished organizing dataset.')

    # filename_train_validation_set_scratch = filename_train_validation_set
    file_path_model = os.path.join(output_path, model_name+str(ii)+'.h5')
    file_path_log = os.path.join(output_path, model_name+str(ii)+'.csv')

    input_dim = (80, 15)

    train_model_validation(filename_train_validation_set=filename_train_validation_set,
                           filename_labels_train_validation_set=filename_labels_train_validation_set,
                           filename_sample_weights=filename_sample_weights,
                           filter_density=1,
                           dropout=0.5,
                           input_shape=input_dim,
                           file_path_model=file_path_model,
                           filename_log=file_path_log,
                           model_name=model_name)

    os.remove(filename_train_validation_set)
    os.remove(filename_labels_train_validation_set)
    os.remove(filename_sample_weights)