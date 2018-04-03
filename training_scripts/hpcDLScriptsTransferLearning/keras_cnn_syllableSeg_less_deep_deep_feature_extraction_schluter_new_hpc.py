import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import finetune_model_validation
import shutil

if __name__ == '__main__':

    nlen = 15
    input_dim = (80, nlen)

    pretrained_model_schluter = '/homedtic/rgong/cnnSyllableSeg/out/schulter_jan_madmom_simpleSampleWeighting_early_stopping_adam_cv_less_deep_0.h5'
    # pretrained_model_schluter = '/Users/gong/Documents/pycharmProjects/jingjuSyllabicSegmentaion/cnnModels/bock/simpleWeighting/schulter_jan_madmom_simpleSampleWeighting_early_stopping_adam_cv_less_deep_0.h5'


    filename_train_validation_set = '/homedtic/rgong/cnnSyllableSeg/syllableSeg/feature_all_artist_filter_madmom.h5'
    filename_labels_train_validation_set = '/homedtic/rgong/cnnSyllableSeg/syllableSeg/labels_train_set_all_syllableSeg_mfccBands2D_old+new_artist_filter_madmom.pickle.gz'
    filename_sample_weights = '/homedtic/rgong/cnnSyllableSeg/syllableSeg/sample_weights_syllableSeg_mfccBands2D_old+new_artist_filter_madmom.pickle.gz'

    tmp_train_validation_set = '/tmp/jingju_pretrained_schluter'
    if not os.path.isdir(tmp_train_validation_set):
        os.mkdir(tmp_train_validation_set)
    filename_temp_train_validation_set = os.path.join(tmp_train_validation_set, 'feature_all_artist_filter_madmom.h5')

    shutil.copy2(filename_train_validation_set, filename_temp_train_validation_set)

    # file_path_model = '../../temp/keras.cnn_syllableSeg_jan_class_weight_mfccBands_2D_all_artist_filter_madmom.h5'
    # file_path_log = '../../temp/keras.cnn_syllableSeg_jan_class_weight_mfccBands_2D_all_artist_filter_madmom.csv'
    #
    # filename_train_validation_set = '/Users/gong/Documents/MTG document/dataset/syllableSeg/feature_all_artist_filter_madmom.h5'
    # filename_labels_train_validation_set = '../../trainingData/labels_train_set_all_syllableSeg_mfccBands2D_old+new_artist_filter_madmom.pickle.gz'
    # filename_sample_weights = '../../trainingData/sample_weights_syllableSeg_mfccBands2D_old+new_artist_filter_madmom.pickle.gz'


    for running_time in range(5):
        # train the final model
        file_path_model = '/homedtic/rgong/cnnSyllableSeg/out/keras.cnn_syllableSeg_jan_artist_filter_less_deep_deep_feature_extraction_schluter'+str(running_time)+'.h5'
        file_path_log = '/homedtic/rgong/cnnSyllableSeg/out/log/keras.cnn_syllableSeg_jan_artist_filter_less_deep_deep_feature_extraction_schluter'+str(running_time)+'.csv'

        finetune_model_validation(filename_train_validation_set=filename_temp_train_validation_set,
                                  filename_labels_train_validation_set=filename_labels_train_validation_set,
                                  filename_sample_weights=filename_sample_weights,
                                  filter_density=1,
                                  dropout=0.5,
                                  input_shape=input_dim,
                                  file_path_model=file_path_model,
                                  filename_log=file_path_log,
                                  model_name='jan_original_pretrained',
                                  path_model=pretrained_model_schluter,
                                  deep=True,
                                  dense=False)