import sys, os, shutil
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preparation_schluter import getTrainingFilenames, concatenateFeatureLabelSampleweights, saveFeatureLabelSampleweights
from models import train_model_validation

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from file_path_bock import *

def syllableSeg_jan_madmom_original_basecode(part, ii, deep=False, dense=False):

    deep_str = 'less_deep_' if deep else ''
    dense_str = 'no_dense_' if not dense else ''

    test_cv_filename = join(bock_cv_path, '8-fold_cv_random_' + str(ii) + '.fold')
    train_fns = getTrainingFilenames(bock_annotations_path, test_cv_filename)
    feature_all, label_all, sample_weights_all, scaler = concatenateFeatureLabelSampleweights(train_fns,
                                                                                              bock_feature_data_path_madmom_simpleSampleWeighting,
                                                                                              n_pattern=15,
                                                                                              nlen=7,
                                                                                              scaling=True)
    filename_train_validation_set = join(bock_feature_data_path_madmom_simpleSampleWeighting, 'temp', 'feature_all_jan_temp_' + str(ii) + '.h5')
    filename_labels_train_validation_set = join(bock_feature_data_path_madmom_simpleSampleWeighting, 'temp', 'labels_train_set_all_jan_temp_' + str(ii) + '.pickle.gz')
    filename_sample_weights = join(bock_feature_data_path_madmom_simpleSampleWeighting, 'temp', 'sample_weights_all_jan_temp_' + str(ii) + '.pickle.gz')
    filename_scaler = join(bock_feature_data_path_madmom_simpleSampleWeighting, 'temp', 'scaler_jan_madmom_simpleSampleWeighting_early_stopping_' + str(ii) + '.pickle.gz')

    saveFeatureLabelSampleweights(feature_all, label_all, sample_weights_all, scaler,
                                  filename_train_validation_set, filename_labels_train_validation_set,
                                  filename_sample_weights, filename_scaler)

    timestamp1 = time.time()
    filename_train_validation_set_scratch = join('/scratch/rgongcnnSyllableSeg_part'+str(part)+'_jan/syllableSeg', 'feature_all_jan_temp_'+str(ii)+'.h5')
    shutil.copy2(filename_train_validation_set, filename_train_validation_set_scratch)
    timestamp2 = time.time()
    print("Copying to scratch took %.2f seconds" % (timestamp2 - timestamp1))

    # train the mode
    file_path_model = '/homedtic/rgong/cnnSyllableSeg/out/schulter_jan_madmom_simpleSampleWeighting_early_stopping_adam_cv_'+dense_str+deep_str+str(ii)+'.h5'
    file_path_log = '/homedtic/rgong/cnnSyllableSeg/out/log/schulter_jan_madmom_simpleSampleWeighting_early_stopping_adam_cv_'+dense_str+deep_str+str(ii)+'.csv'

    # filename_train_validation_set_scratch = filename_train_validation_set
    # file_path_model = '../../temp/schulter_jan_madmom_simpleSampleWeighting_cv_'+str(ii)+'.h5'
    # file_path_log = '../../temp/schulter_jan_madmom_simpleSampleWeighting_cv_'+str(ii)+'.csv'

    input_dim = (80, 15)

    train_model_validation(filename_train_validation_set=filename_train_validation_set_scratch,
                            filename_labels_train_validation_set=filename_labels_train_validation_set,
                            filename_sample_weights=filename_sample_weights,
                            filter_density=1,
                            dropout=0.5,
                           input_shape=input_dim,
                            file_path_model=file_path_model,
                           filename_log=file_path_log,
                           deep=deep,
                           dense=dense)

    os.remove(filename_train_validation_set)
    os.remove(filename_labels_train_validation_set)
    os.remove(filename_sample_weights)