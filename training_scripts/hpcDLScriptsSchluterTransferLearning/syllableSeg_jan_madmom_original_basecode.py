import sys, os, shutil
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preparation_schluter import getTrainingFilenames
from data_preparation_schluter import concatenateFeatureLabelSampleweights
from data_preparation_schluter import saveFeatureLabelSampleweights
from data_preparation_schluter import concatenateFeatureLabelSampleweightsJingju
from models import train_model_validation

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from file_path_bock import *

def syllableSeg_jan_madmom_original_basecode(part, ii, deep=False, dense=False):

    deep_str = 'less_deep_' if deep else ''
    dense_str = 'no_dense_' if not dense else ''

    test_cv_filename = join(bock_cv_path, '8-fold_cv_random_' + str(ii) + '.fold')
    train_fns = getTrainingFilenames(bock_annotations_path, test_cv_filename)
    feature_schluter, label_schluter, sample_weights_schluter, _ = concatenateFeatureLabelSampleweights(train_fns,
                                                                                                        bock_feature_data_path_madmom_simpleSampleWeighting,
                                                                                                        n_pattern=15,
                                                                                                        nlen=7,
                                                                                                        scaling=None)

    # artist training dataset
    # filename_feature_jingju = os.path.join('/scratch/rgongcnnSyllableSeg_part'+str(part)+'_jan/syllableSeg' ,'feature_all_artist_filter_madmom.h5')
    # filename_labels_jingju = '/homedtic/rgong/cnnSyllableSeg/syllableSeg/labels_train_set_all_syllableSeg_mfccBands2D_old+new_artist_filter_madmom.pickle.gz'
    # filename_sample_weights_jingju = '/homedtic/rgong/cnnSyllableSeg/syllableSeg/sample_weights_syllableSeg_mfccBands2D_old+new_artist_filter_madmom.pickle.gz'

    filename_feature_jingju = '/Users/gong/Documents/MTG document/dataset/syllableSeg/feature_all_artist_filter_madmom.h5'
    filename_labels_jingju = '../../trainingData/labels_train_set_all_syllableSeg_mfccBands2D_old+new_artist_filter_madmom.pickle.gz'
    filename_sample_weights_jingju = '../../trainingData/sample_weights_syllableSeg_mfccBands2D_old+new_artist_filter_madmom.pickle.gz'

    # combine jingju and bock dataset
    feature_all, label_all, sample_weights_all, scaler = \
        concatenateFeatureLabelSampleweightsJingju(feature_schluter=feature_schluter,
                                                   label_schluter=label_schluter,
                                                   sample_weights_schluter=sample_weights_schluter,
                                                   filename_jingju_features=filename_feature_jingju,
                                                   filename_jingju_labels=filename_labels_jingju,
                                                   filename_jingju_sample_weights=filename_sample_weights_jingju,
                                                   scaling=True)

    filename_train_validation_set = join(bock_feature_data_path_madmom_simpleSampleWeighting, 'temp', 'feature_all_jan_temp_' + str(ii) + '.h5')
    filename_labels_train_validation_set = join(bock_feature_data_path_madmom_simpleSampleWeighting, 'temp', 'labels_train_set_all_jan_temp_' + str(ii) + '.pickle.gz')
    filename_sample_weights = join(bock_feature_data_path_madmom_simpleSampleWeighting, 'temp', 'sample_weights_all_jan_temp_' + str(ii) + '.pickle.gz')
    filename_scaler = join(bock_feature_data_path_madmom_simpleSampleWeighting, 'temp', 'scaler_jan_madmom_simpleSampleWeighting_early_stopping_schluter_jingju_dataset_' + str(ii) + '.pickle.gz')

    saveFeatureLabelSampleweights(feature_all, label_all, sample_weights_all, scaler,
                                  filename_train_validation_set, filename_labels_train_validation_set,
                                  filename_sample_weights, filename_scaler)

    # timestamp1 = time.time()
    # filename_train_validation_set_scratch = join('/scratch/rgongcnnSyllableSeg_part'+str(part)+'_jan/syllableSeg', 'feature_all_jan_temp_'+str(ii)+'.h5')
    # shutil.copy2(filename_train_validation_set, filename_train_validation_set_scratch)
    # timestamp2 = time.time()
    # print("Copying to scratch took %.2f seconds" % (timestamp2 - timestamp1))

    # # train the mode
    # file_path_model = '/homedtic/rgong/cnnSyllableSeg/out/schulter_jan_madmom_simpleSampleWeighting_early_stopping_adam_cv_jingju_'+dense_str+deep_str+str(ii)+'.h5'
    # file_path_log = '/homedtic/rgong/cnnSyllableSeg/out/log/schulter_jan_madmom_simpleSampleWeighting_early_stopping_adam_cv_jingju_'+dense_str+deep_str+str(ii)+'.csv'

    filename_train_validation_set_scratch = filename_train_validation_set
    file_path_model = '../../temp/schulter_jan_madmom_simpleSampleWeighting_cv_'+str(ii)+'.h5'
    file_path_log = '../../temp/schulter_jan_madmom_simpleSampleWeighting_cv_'+str(ii)+'.csv'
    filename_train_validation_set_scratch = filename_train_validation_set

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

if __name__ == '__main__':
    syllableSeg_jan_madmom_original_basecode(1, 0, deep=True, dense=False)