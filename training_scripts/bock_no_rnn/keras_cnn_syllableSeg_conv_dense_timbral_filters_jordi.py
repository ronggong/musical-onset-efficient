
# from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import sys, os
import time
import shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preparation import load_data_jingju
from data_preparation_schluter import getTrainingFilenames, concatenateFeatureLabelSampleweights, saveFeatureLabelSampleweights
from models import jordi_model, model_train

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from file_path_bock import *

nlen = 21
input_dim = (80, nlen)


def train_model(filename_train_validation_set,
                filename_labels_train_validation_set,
                filename_sample_weights,
                filter_density_1, filter_density_2,
                pool_n_row, pool_n_col,
                dropout, input_shape,
                file_path_model, filename_log):
    """
    train final model save to model path
    """

    filenames_train, Y_train, sample_weights_train, \
    filenames_validation, Y_validation, sample_weights_validation, \
    filenames_features, Y_train_validation, sample_weights, class_weights = \
        load_data_jingju(filename_labels_train_validation_set,
                         filename_sample_weights)


    model_0 = jordi_model(filter_density_1, filter_density_2,
                            pool_n_row, pool_n_col,
                            dropout, input_shape,
                          'timbral')

    batch_size = 128
    patience = 10

    print(model_0.count_params())

    model_train(model_0, batch_size, patience, input_shape,
                filename_train_validation_set,
                filenames_train, Y_train, sample_weights_train,
                filenames_validation, Y_validation, sample_weights_validation,
                filenames_features, Y_train_validation, sample_weights, class_weights,
                file_path_model, filename_log)


if __name__ == '__main__':
    for ii in range(8):
        test_cv_filename = join(bock_cv_path, '8-fold_cv_random_' + str(ii) + '.fold')
        train_fns = getTrainingFilenames(bock_annotations_path, test_cv_filename)
        feature_all, label_all, sample_weights_all, scaler = concatenateFeatureLabelSampleweights(train_fns, schluter_feature_data_path)

        filename_train_validation_set = join(schluter_feature_data_path, 'temp', 'feature_all_timbral_temp.h5')
        filename_labels_train_validation_set = join(schluter_feature_data_path, 'temp', 'labels_train_set_all_timbral_temp.pickle.gz')
        filename_sample_weights = join(schluter_feature_data_path, 'temp', 'sample_weights_all_timbral_temp.pickle.gz')
        filename_scaler = join(schluter_feature_data_path, 'temp', 'scaler_timbral_'+str(ii)+'.pickle.gz')

        saveFeatureLabelSampleweights(feature_all, label_all, sample_weights_all, scaler,
                                      filename_train_validation_set, filename_labels_train_validation_set, filename_sample_weights, filename_scaler)

        # copy feature to scratch
        timestamp1 = time.time()
        filename_train_validation_set_scratch = join('/scratch/rgongcnnSyllableSeg_timbral/syllableSeg/',
                                                     'feature_all_timbral_temp.h5')
        shutil.copy2(filename_train_validation_set, filename_train_validation_set_scratch)
        timestamp2 = time.time()
        print("Copying to scratch took %.2f seconds" % (timestamp2 - timestamp1))

        file_path_model = '/homedtic/rgong/cnnSyllableSeg/out/schulter_timbral_cv_'+str(ii)+'.h5'
        file_path_log = '/homedtic/rgong/cnnSyllableSeg/out/log/schulter_timbral_cv_'+str(ii)+'.csv'

        train_model(filename_train_validation_set=filename_train_validation_set_scratch,
                    filename_labels_train_validation_set=filename_labels_train_validation_set,
                    filename_sample_weights=filename_sample_weights,
                    filter_density_1=1, filter_density_2=1,
                    pool_n_row=5, pool_n_col=3,
                    dropout=0.3, input_shape=input_dim,
                    file_path_model=file_path_model, filename_log=file_path_log)

        os.remove(filename_train_validation_set)
        os.remove(filename_labels_train_validation_set)
        os.remove(filename_sample_weights)
