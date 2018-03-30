from os.path import join
from deprecated_parameters import mth_ODF
from deprecated_parameters import fusion
from deprecated_parameters import layer2
from deprecated_parameters import filter_shape
from deprecated_parameters import varin

cnnModels_path = ''
root_path = ''

# acoustic model name and path
filename_scaler_am = 'scaler_syllableSeg_am_mfccBands2D.pkl'
full_path_mfccBands_2D_scaler_am = join(cnnModels_path, 'am', filename_scaler_am)
filename_keras_cnn_am = 'keras.cnn_syllableSeg_am_mfccBands_2D_all_optim.h5'
full_path_keras_cnn_am = join(cnnModels_path, 'am', filename_keras_cnn_am)

# score paths
score_path = '/Users/gong/Documents/MTG document/Jingju arias/Scores/artistAlbumFilterAudioScoreAlignment'
score_info_filename = '0.Lyrics.csv'
score2midi_path = join(root_path, 'audioScoreAlignment', 'midi')
midi2wav_path = join(root_path, 'audioScoreAlignment', 'wav')

# riyaz dataset
riyaz_dataset_root_path = '/Users/gong/Documents/MTG document/dataset/riyaz_corrected_data'

riyaz_mp3_path = join(riyaz_dataset_root_path, 'mp3')
riyaz_groundtruthlab_path = join(riyaz_dataset_root_path, 'groundtruth')
riyaz_score_path = join(riyaz_dataset_root_path, 'score')
riyaz_segPhrase_path = join(riyaz_dataset_root_path, 'segPhrase')
riyaz_feature_data_path = riyaz_dataset_root_path

if mth_ODF == 'jan':
    if varin['dataset'] == 'ismir':
        # filename_keras_cnn_0  = 'keras.cnn_syllableSeg_jan_class_weight_mfccBands_2D_all_ismir_madmom_early_stopping'
        # cnnModel_name         = 'jan_old+new_ismir_madmom_early_stopping'
        # filename_keras_cnn_0 = 'keras.cnn_syllableSeg_jan_ismir_madmom_early_stopping_deep'
        # cnnModel_name = 'jan_ismir_madmom_early_stopping_deep'
        # filename_keras_cnn_0 = 'keras.cnn_syllableSeg_jan_ismir_batchNorm'
        # cnnModel_name = 'jan_ismir_batchNorm'
        # filename_keras_cnn_0 = 'keras.cnn_syllableSeg_jan_ismir_madmom_early_stopping_relu'
        # cnnModel_name = 'jan_ismir_madmom_early_stopping_relu'
        # filename_keras_cnn_0 = 'keras.cnn_syllableSeg_jan_ismir_madmom_early_stopping_no_dense'
        # cnnModel_name = 'jan_ismir_madmom_early_stopping_no_dense'
        filename_keras_cnn_0 = 'keras.cnn_syllableSeg_jan_ismir_less_deep'
        cnnModel_name = 'jan_ismir_less_deep'
    else:
        # filename_keras_cnn_0  = 'keras.cnn_syllableSeg_jan_class_weight_mfccBands_2D_all_artist_filter_madmom_early_stopping'
        # cnnModel_name         = 'jan_old+new_artist_filter_madmom_early_stopping'
        # filename_keras_cnn_0 = 'keras.cnn_syllableSeg_jan_artist_filter_madmom_early_stopping_relu'
        # cnnModel_name = 'jan_artist_filter_madmom_early_stopping_relu'
        # filename_keras_cnn_0 = 'keras.cnn_syllableSeg_jan_artist_filter_madmom_early_stopping_deep'
        # cnnModel_name = 'jan_artist_filter_madmom_early_stopping_deep'
        # filename_keras_cnn_0 = 'keras.cnn_syllableSeg_jan_artist_filter_madmom_early_stopping_no_dense'
        # cnnModel_name = 'jan_artist_filter_madmom_early_stopping_no_dense'
        # filename_keras_cnn_0 = 'keras.cnn_syllableSeg_jan_artist_filter_less_deep'
        # cnnModel_name = 'jan_artist_filter_less_deep'
        # filename_keras_cnn_0 = 'keras.cnn_syllableSeg_jan_artist_filter_deep_strong_front'
        # cnnModel_name = 'jan_artist_filter_deep_strong_front'
        filename_keras_cnn_0 = 'jan_joint_subset_series'
        cnnModel_name = 'jan_artist_filter_joint_subset_series'
        # TODO use schluter to evaluate jingju
        # filename_keras_cnn_0 = 'keras.cnn_syllableSeg_jan_artist_filter_less_deep_deep_feature_extraction_schluter'
        # cnnModel_name = 'jan_artist_filter_less_deep_deep_feature_extraction_schluter'

elif mth_ODF == 'jan_chan3':
    filename_keras_cnn_0  = 'keras.cnn_syllableSeg_jan_class_weight_3_chans_mfccBands_2D_all_optim.h5'
    # filename_keras_cnn_0  = 'keras.cnn_syllableSeg_jan_class_weight_3_chans_layer1_70_mfccBands_2D_all_optim.h5'
    # filename_keras_cnn_0  = 'keras.cnn_syllableSeg_jan_no_class_weight_mfccBands_2D_all_optim.h5'
    cnnModel_name = 'jan_cw_3_chans'
elif mth_ODF == 'jordi_horizontal_timbral':
    if layer2 == 20:
        filename_keras_cnn_0 = 'keras.cnn_syllableSeg_jordi_class_weight_with_conv_dense_horizontal_timbral_filter_layer2_20_mfccBands_2D_all_optim.h5'
        cnnModel_name = 'jordi_cw_conv_dense_horizontal_timbral_filter_layer2_20'
    else:
        filename_keras_cnn_0 = 'keras.cnn_syllableSeg_jordi_class_weight_with_conv_dense_horizontal_timbral_filter_mfccBands_2D_all_optim.h5'
        cnnModel_name = 'jordi_cw_conv_dense_horizontal_timbral_filter'
else:
    # mth_ODF == 'jordi'
    if fusion:
        if layer2 == 20:
            filename_keras_cnn_0 = 'keras.cnn_syllableSeg_jordi_temporal_mfccBands_2D_all_ismir_madmom.h5'
            filename_keras_cnn_1 = 'keras.cnn_syllableSeg_jordi_timbral_mfccBands_2D_all_ismir_madmom.h5'
            cnnModel_name = 'jordi_fusion_ismir_madmom'
        else:
            filename_keras_cnn_0 = 'keras.cnn_syllableSeg_jan_class_weight_mfccBands_2D_all_old+new.h5'
            filename_keras_cnn_1 = 'keras.cnn_syllableSeg_jordi_temporal_class_weight_with_conv_dense_mfccBands_2D_old+new.h5'
            filename_keras_cnn_2 = 'keras.cnn_syllableSeg_jordi_timbral_class_weight_with_conv_dense_filter_mfccBands_2D_old+new.h5'
            cnnModel_name = 'jordi_fusion_old+new_artist_split'
    else:
        if filter_shape == 'temporal':
            # filename_keras_cnn_0  = 'keras.cnn_syllableSeg_jordi_class_weight_mfccBands_2D_all_optim.h5'
            # filename_keras_cnn_0  = 'keras.cnn_syllableSeg_jordi_class_weight_with_dense_mfccBands_2D_all_optim.h5'
            if layer2 == 20:

                if varin['dataset'] == 'ismir':
                    # filename_keras_cnn_0 = 'keras.cnn_syllableSeg_jordi_temporal_mfccBands_2D_all_ismir_madmom_early_stopping_more_params'
                    # cnnModel_name        = 'jordi_temporal_ismir_madmom_early_stopping_more_params'

                    filename_keras_cnn_0 = 'keras.cnn_syllableSeg_jordi_temporal_mfccBands_2D_all_ismir_madmom_early_stopping_jan_params'
                    cnnModel_name = 'jordi_temporal_ismir_madmom_early_stopping_jan_params'
                else:
                    # filename_keras_cnn_0  = 'keras.cnn_syllableSeg_jordi_temporal_mfccBands_2D_all_artist_filter_madmom_early_stopping_more_params'
                    # cnnModel_name         = 'jordi_temporal_artist_filter_madmom_early_stopping_more_params'

                    filename_keras_cnn_0 = 'keras.cnn_syllableSeg_jordi_temporal_mfccBands_2D_all_artist_filter_madmom_early_stopping_jan_params'
                    cnnModel_name = 'jordi_temporal_artist_filter_madmom_early_stopping_jan_params'


            else:
                # layer2 32 nodes
                filename_keras_cnn_0 = 'keras.cnn_syllableSeg_temporal_class_weight_mfccBands_2D_all_riyaz.h5'
                # filename_keras_cnn_0  = 'keras.cnn_syllableSeg_jordi_class_weight_with_conv_dense_149k_mfccBands_2D_all_optim.h5'
                # filename_keras_cnn_1  = 'keras.cnn_syllableSeg_jordi_class_weight_with_conv_dense_second_model_32_mfccBands_2D_all_optim.h5'
                cnnModel_name = 'jordi_temporal_old+new_artist_filter_split'

        else:
            # timbral filter shape
            if layer2 == 20:
                if varin['dataset'] == 'ismir':
                    filename_keras_cnn_0 = 'keras.cnn_syllableSeg_jordi_timbral_mfccBands_2D_all_ismir_madmom_early_stopping_more_params'
                    cnnModel_name = 'jordi_timbral_ismir_madmom_early_stopping_more_params'
                else:
                    filename_keras_cnn_0 = 'keras.cnn_syllableSeg_jordi_timbral_mfccBands_2D_all_artist_filter_madmom_early_stopping_more_params'
                    cnnModel_name = 'jordi_timbral_artist_filter_madmom_early_stopping_more_params'

            else:
                # layer2 32 nodes
                filename_keras_cnn_0 = 'keras.cnn_syllableSeg_jordi_timbral_class_weight_with_conv_dense_filter_mfccBands_2D_artist_filter_split_2_train.h5'
                # filename_keras_cnn_0  = 'keras.cnn_syllableSeg_jordi_timbral_class_weight_with_conv_dense_filter_mfccBands_2D_artist_split.h5'

                # filename_keras_cnn_0 = 'keras.cnn_syllableSeg_jordi_class_weight_with_conv_dense_timbral_filter_152k_mfccBands_2D_all_optim.h5'
                # filename_keras_cnn_1 = 'keras.cnn_syllableSeg_jordi_class_weight_with_conv_dense_timbral_filter_second_model_32_mfccBands_2D_all_optim.h5'
                cnnModel_name = 'jordi_timbral_old+new_artist_filter_split_2_train'


if varin['dataset'] == 'ismir':
    filename_scaler_onset = 'scaler_syllable_mfccBands2D_old+new_ismir_madmom.pkl'
else:
    if 'joint' in filename_keras_cnn_0:
        filename_scaler_onset = 'scaler_joint_subset.pkl'
    else:
        # TODO use schluter to evaluate jingju
        filename_scaler_onset = 'scaler_jan_no_rnn.pkl'
        # filename_scaler_onset = 'scaler_jan_madmom_simpleSampleWeighting_early_stopping_schluter_jingju_dataset_'

# CRNN non-overlapping path
cnnModel_name = varin['dataset']+'_jan_madmom_simpleSampleWeighting_early_stopping_adam_cv_phrase'
