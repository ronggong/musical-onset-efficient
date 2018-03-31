from os.path import join
from os.path import dirname
from parameters_schluter import varin
from file_path_shared import feature_data_path


root_path = join(dirname(__file__), '..')

weighting_str = \
    'simpleSampleWeighting' if varin['sample_weighting'] == 'simpleWeighting' else 'positiveThreeSampleWeighting'

bock_dataset_root_path = '/Users/ronggong/Documents_using/MTG document/dataset/onsets'

bock_audio_path = join(bock_dataset_root_path, 'audio')

bock_cv_path = join(bock_dataset_root_path, 'splits')

bock_annotations_path = join(bock_dataset_root_path, 'annotations')

bock_feature_data_path_madmom_simpleSampleWeighting = \
    join(feature_data_path, 'bock_simpleSampleWeighting')

bock_feature_data_path_madmom_simpleSampleWeighting_3channel = \
    join(feature_data_path, 'bock_simpleSampleWeighting_3channel')

bock_feature_data_path_madmom_complicateSampleWeighting = \
    join(feature_data_path, 'bock_complicateSampleWeighting')

bock_feature_data_path_madmom_positiveThreeSampleWeighting = \
    join(feature_data_path, 'bock_postiveThreeSampleWeighting')

bock_feature_data_path_madmom_simpleSampleWeighting_phrase = \
    join(feature_data_path, 'bock_simpleSampleWeighting_phrase')

bock_cnn_model_path = join(root_path, 'pretrained_models', 'bock', varin['sample_weighting'])

scaler_bock_phrase_model_path = join(bock_cnn_model_path, 'scaler_bock_phrase.pkl')

detection_results_path = join(root_path, 'eval', 'results')

bock_results_path = join(root_path, 'eval', 'bock', 'results')

# jingju model
jingju_cnn_model_path = join(root_path, 'pretrained_models', 'jingju', varin['sample_weighting'])

full_path_jingju_scaler = join(jingju_cnn_model_path, 'scaler_jan_no_rnn.pkl')
