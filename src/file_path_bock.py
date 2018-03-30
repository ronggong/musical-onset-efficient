from os.path import join
from os.path import dirname
from parameters_schluter import varin
from file_path_shared import feature_data_path

if varin['architecture'] == 'jan_no_dense':
    filter_shape_0, phrase_eval, overlap, bidi, relu, deep, less_deep, no_dense = \
        'jan', False, False, False, False, False, False, True
elif varin['architecture'] == 'jan_sigmoid':
    filter_shape_0, phrase_eval, overlap, bidi, relu, deep, less_deep, no_dense = \
        'jan', False, False, False, False, False, False, False
elif varin['architecture'] == 'jan_relu':
    filter_shape_0, phrase_eval, overlap, bidi, relu, deep, less_deep, no_dense = \
        'jan', False, False, False, True, False, False, False
elif varin['architecture'] == 'jan_deep':
    filter_shape_0, phrase_eval, overlap, bidi, relu, deep, less_deep, no_dense = \
        'jan', False, False, False, False, True, False, False
elif varin['architecture'] == 'jan_less_deep':
    filter_shape_0, phrase_eval, overlap, bidi, relu, deep, less_deep, no_dense = \
        'jan', False, False, False, False, False, True, False
elif varin['architecture'] == 'jordi_temporal':
    filter_shape_0, phrase_eval, overlap, bidi, relu, deep, less_deep, no_dense = \
        'jordi_temporal_schluter', False, False, False, False, False, False, False
elif varin['architecture'] == 'jan_bidi_100':
    len_seq = 100  # sub-sequence length
elif varin['architecture'] == 'jan_bidi_200':
    len_seq = 200
elif varin['architecture'] == 'jan_bidi_400':
    len_seq = 400
elif varin['architecture'] == 'transfer_pretrained' or \
                varin['architecture'] == 'transfer_2_layers_feature' or \
                varin['architecture'] == 'transfer_deep_feature':
    filter_shape_0, phrase_eval, overlap, bidi, relu, deep, less_deep, no_dense = \
        'jan', False, False, False, False, False, True, True
else:
    raise ValueError('There is no such architecture %s.' % varin['architecture'])

if 'jan_bidi' in varin['architecture']:
    filter_shape_0, phrase_eval, overlap, bidi, relu, deep, less_deep, no_dense = \
        'jan', True, True, True, False, False, False, False

overlap_str = '_overlap' if overlap else ''
phrase_str = '_phrase' if phrase_eval else ''
bidi_str = '_bidi_100' if bidi else ''
relu_str = '_relu' if relu else ''

if deep:
    deep_str = '_deep'
elif less_deep:
    deep_str = '_less_deep'
else:
    deep_str = ''

if no_dense:
    if varin['architecture'] == 'transfer_pretrained':
        no_dense_str = '_pretrained_jingju_no_dense'
    elif varin['architecture'] == 'transfer_2_layers_feature':
        no_dense_str = '_feature_extraction_jingju_no_dense'
    elif varin['architecture'] == 'transfer_deep_feature':
        no_dense_str = '_deep_feature_extraction_jingju_no_dense'
    else:
        no_dense_str = '_no_dense'
else:
    no_dense_str = ''

root_path = join(dirname(__file__), '..')

weighting_str = \
    'simpleSampleWeighting' if varin['sample_weighting'] == 'simpleWeighting' else 'positiveThreeSampleWeighting'

# schluter_dataset_root_path = '/Users/gong/Documents/MTG document/dataset/onsets'

# schluter_dataset_root_path = '/datasets/MTG/projects/compmusic/jingju_datasets/schluter/'

schluter_dataset_root_path = '/media/gong/ec990efa-9ee0-4693-984b-29372dcea0d1/Data/RongGong/onsets'

schluter_audio_path = join(schluter_dataset_root_path, 'audio')

schluter_cv_path = join(schluter_dataset_root_path, 'splits')

schluter_annotations_path = join(schluter_dataset_root_path, 'annotations')


schluter_feature_data_path_madmom_simpleSampleWeighting = join(feature_data_path, 'bock_simpleSampleWeighting')

schluter_feature_data_path_madmom_simpleSampleWeighting_3channel = join(feature_data_path, 'bock_simpleSampleWeighting_3channel')

schluter_feature_data_path_madmom_complicateSampleWeighting = join(feature_data_path, 'bock_complicateSampleWeighting')

schluter_feature_data_path_madmom_positiveThreeSampleWeighting = join(feature_data_path, 'bock_postiveThreeSampleWeighting')

schluter_feature_data_path_madmom_simpleSampleWeighting_phrase = join(feature_data_path, 'bock_simpleSampleWeighting_phrase')

schluter_cnn_model_path = join(root_path, 'cnnModels', 'schluter', varin['sample_weighting'])

scaler_schluter_phrase_model_path = join(schluter_cnn_model_path,
                                         'scaler_syllable_mfccBands2D_schluter_madmom_phrase.pkl')

eval_results_path = join(root_path, 'eval', 'results')

schluter_results_path = join(root_path, 'eval', 'schluter', 'results')

jingju_cnn_model_path = join(root_path, 'cnnModels', 'jingju', varin['sample_weighting'])

full_path_mfccBands_2D_scaler_onset = \
    join(jingju_cnn_model_path, 'scaler_jan_no_rnn.pkl')
