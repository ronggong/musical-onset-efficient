import os
import sys
import gzip
import pickle
import cPickle
import numpy as np
from keras.models import load_model

sys.path.append(os.path.join(os.path.dirname(__file__), "./src/"))

from experiment_process_helper import odf_calculation_crnn
from experiment_process_helper import boundary_decoding
from audio_preprocessing import getMFCCBands2DMadmom
from utilFunctions import featureReshape
from utilFunctions import smooth_obs
from training_scripts.models_CRNN import jan_original
from parameters_jingju import varin
from plot_code import plot_jingju_odf_colab

import pyximport
pyximport.install(reload_support=True,
                  setup_args={'include_dirs': np.get_include()})

# import peak-picking module
from madmom.features.onsets import OnsetPeakPickingProcessor
# import score-informed HMM module
import viterbiDecoding


def feature_preprocessing(feature, scaler):
    # scale feature
    feature = scaler.transform(feature)
    # reshape feature
    feature = featureReshape(feature, nlen=7)
    feature = np.expand_dims(feature, axis=1)

    return feature


def odf_postprocessing(odf):
    odf = np.squeeze(odf)
    odf = smooth_obs(odf)
    return odf


# parameters
hopsize_t = 0.01
fs = 44100
threshold_peak_picking = 0.20

# pretrained model path
path_models_jingju = './pretrained_models/jingju/simpleWeighting'
path_models_bock = './pretrained_models/bock/simpleWeighting'

wav_jingju = './inputs/audio_5.wav'

# load scaler
path_scaler_no_rnn = os.path.join(path_models_jingju, 'scaler_jan_no_rnn.pkl')
scaler_no_rnn = pickle.load(open(path_scaler_no_rnn))

path_scaler_crnn = os.path.join(path_models_jingju, 'scaler_jingju_crnn_phrase.pkl')
scaler_crnn = pickle.load(open(path_scaler_crnn))

path_scaler_bock = os.path.join(path_models_bock, 'scaler_bock_0.pickle.gz')
scaler_bock = cPickle.load(gzip.open(path_scaler_bock))

# load score
boundary_groundtruth, syllable_duration = pickle.load(open('./inputs/input_jingju_5.pkl', 'r'))

# load audio
log_mel = getMFCCBands2DMadmom(wav_jingju, fs, hopsize_t, channel=1)

log_mel_jingju_no_rnn = feature_preprocessing(log_mel, scaler_no_rnn)
log_mel_jingju_crnn = scaler_crnn.transform(log_mel)
log_mel_bock = feature_preprocessing(log_mel, scaler_bock)

# baseline
print('computing baseline ODF ...')
model_baseline = load_model(os.path.join(path_models_jingju, 'baseline0.h5'))
odf_baseline = model_baseline.predict(log_mel_jingju_no_rnn, batch_size=128, verbose=2)
odf_baseline = odf_postprocessing(odf_baseline)

# relu dense
print('computing relu dense ODF ...')
model_relu_dense = load_model(os.path.join(path_models_jingju, 'relu_dense0.h5'))
odf_relu_dense = model_relu_dense.predict(log_mel_jingju_no_rnn, batch_size=128, verbose=2)
odf_relu_dense = odf_postprocessing(odf_relu_dense)

# no dense
print('computing no dense ODF ...')
model_no_dense = load_model(os.path.join(path_models_jingju, 'no_dense0.h5'))
odf_no_dense = model_no_dense.predict(log_mel_jingju_no_rnn, batch_size=128, verbose=2)
odf_no_dense = odf_postprocessing(odf_no_dense)

# temporal
print('computing temporal ODF ...')
model_temporal = load_model(os.path.join(path_models_jingju, 'temporal0.h5'))
odf_temporal = model_temporal.predict(log_mel_jingju_no_rnn, batch_size=128, verbose=2)
odf_temporal = odf_postprocessing(odf_temporal)

# bidi lstms 100
print('computing bidi lstms 100 ODF ...')
# initialize the model
model_bidi_lstms_100 = jan_original(filter_density=1,
                                    dropout=0.5,
                                    input_shape=(1, 100, 1, 80, 15),
                                    batchNorm=False,
                                    dense_activation='sigmoid',
                                    channel=1,
                                    stateful=False,
                                    training=False,
                                    bidi=True)

# load the model weights
model_bidi_lstms_100.load_weights(os.path.join(path_models_jingju, 'bidi_lstms_1000.h5'))
# calculate odf
odf_bidi_lstms_100, _ = odf_calculation_crnn(log_mel_jingju_crnn,
                                             log_mel_jingju_crnn,
                                             model_bidi_lstms_100,
                                             frame_start=0,
                                             frame_end=len(log_mel),
                                             len_seq=100,
                                             stateful=False)
odf_bidi_lstms_100 = odf_postprocessing(odf_bidi_lstms_100)

# bidi lstms 200
print('computing bidi lstms 200 ODF ...')
# initialize the model
model_bidi_lstms_200 = jan_original(filter_density=1,
                                    dropout=0.5,
                                    input_shape=(1, 200, 1, 80, 15),
                                    batchNorm=False,
                                    dense_activation='sigmoid',
                                    channel=1,
                                    stateful=False,
                                    training=False,
                                    bidi=True)

# load the model weights
model_bidi_lstms_200.load_weights(os.path.join(path_models_jingju, 'bidi_lstms_2000.h5'))
# calculate odf
odf_bidi_lstms_200, _ = odf_calculation_crnn(log_mel_jingju_crnn,
                                             log_mel_jingju_crnn,
                                             model_bidi_lstms_200,
                                             frame_start=0,
                                             frame_end=len(log_mel),
                                             len_seq=200,
                                             stateful=False)
odf_bidi_lstms_200 = odf_postprocessing(odf_bidi_lstms_200)

# bidi lstms 400
print('computing bidi lstms 400 ODF ...')
# initialize the model
model_bidi_lstms_400 = jan_original(filter_density=1,
                                    dropout=0.5,
                                    input_shape=(1, 400, 1, 80, 15),
                                    batchNorm=False,
                                    dense_activation='sigmoid',
                                    channel=1,
                                    stateful=False,
                                    training=False,
                                    bidi=True)

# load the model weights
model_bidi_lstms_400.load_weights(os.path.join(path_models_jingju, 'bidi_lstms_4000.h5'))
# calculate odf
odf_bidi_lstms_400, _ = odf_calculation_crnn(log_mel_jingju_crnn,
                                             log_mel_jingju_crnn,
                                             model_bidi_lstms_400,
                                             frame_start=0,
                                             frame_end=len(log_mel),
                                             len_seq=400,
                                             stateful=False)
odf_bidi_lstms_400 = odf_postprocessing(odf_bidi_lstms_400)

# 9 layers cnn
print('computing 9 layers CNN ODF ...')
model_9_layers_cnn = load_model(os.path.join(path_models_jingju, '9_layers_cnn0.h5'))
odf_9_layers_cnn = model_9_layers_cnn.predict(log_mel_jingju_no_rnn, batch_size=128, verbose=2)
odf_9_layers_cnn = odf_postprocessing(odf_9_layers_cnn)

# 5 layers cnn
print('computing 5 layers CNN ODF ...')
model_5_layers_cnn = load_model(os.path.join(path_models_jingju, '5_layers_cnn0.h5'))
odf_5_layers_cnn = model_5_layers_cnn.predict(log_mel_jingju_no_rnn, batch_size=128, verbose=2)
odf_5_layers_cnn = odf_postprocessing(odf_5_layers_cnn)

# pretrained
print('computing pretrained transfer learning ODF ...')
model_pretrained = load_model(os.path.join(path_models_bock, '5_layers_cnn0.h5'))
odf_pretrained = model_pretrained.predict(log_mel_bock, batch_size=128, verbose=2)
odf_pretrained = odf_postprocessing(odf_pretrained)

# retrained
print('computing retrained transfer learning ODF ...')
model_retrained = load_model(os.path.join(path_models_jingju, 'retrained0.h5'))
odf_retrained = model_retrained.predict(log_mel_jingju_no_rnn, batch_size=128, verbose=2)
odf_retrained = odf_postprocessing(odf_retrained)

# feature extractor a
print('computing feature extractor a transfer learning ODF ...')
model_feature_extractor_a = load_model(os.path.join(path_models_jingju, 'feature_extractor_a0.h5'))
odf_feature_extractor_a = model_feature_extractor_a.predict([log_mel_jingju_no_rnn, log_mel_jingju_no_rnn], batch_size=128, verbose=2)
odf_feature_extractor_a = odf_postprocessing(odf_feature_extractor_a)

# feature extractor b
print('computing feature extractor b transfer learning ODF ...')
model_feature_extractor_b = load_model(os.path.join(path_models_jingju, 'feature_extractor_b0.h5'))
odf_feature_extractor_b = model_feature_extractor_b.predict([log_mel_jingju_no_rnn, log_mel_jingju_no_rnn], batch_size=128, verbose=2)
odf_feature_extractor_b = odf_postprocessing(odf_feature_extractor_b)


list_odfs = [odf_baseline, odf_relu_dense, odf_no_dense, odf_temporal, odf_bidi_lstms_100, odf_bidi_lstms_200,
             odf_bidi_lstms_400, odf_9_layers_cnn, odf_5_layers_cnn, odf_pretrained, odf_retrained,
             odf_feature_extractor_a, odf_feature_extractor_b]

# plot ODFs, red lines in the mel bands are ground truth boundaries
plot_jingju_odf_colab(log_mel=log_mel,
                      hopsize_t=hopsize_t,
                      list_odfs=list_odfs,
                      groundtruth=boundary_groundtruth)


# define a function for unifying two onset selection method
def onset_selection(list_odfs,
                    syllable_duration=None,
                    method='peakPicking',
                    threshold=None,
                    hopsize_t=0.01,
                    viterbiDecoding=None,
                    OnsetPeakPickingProcessor=None,
                    varin=None):

    odf_baseline, odf_relu_dense, odf_no_dense, odf_temporal, odf_bidi_lstms_100, odf_bidi_lstms_200, \
    odf_bidi_lstms_400, odf_9_layers_cnn, odf_5_layers_cnn, odf_pretrained, odf_retrained, \
    odf_feature_extractor_a, odf_feature_extractor_b = list_odfs[0], \
                                                       list_odfs[1], \
                                                       list_odfs[2], \
                                                       list_odfs[3], \
                                                       list_odfs[4], \
                                                       list_odfs[5], \
                                                       list_odfs[6], \
                                                       list_odfs[7], \
                                                       list_odfs[8], \
                                                       list_odfs[9], \
                                                       list_odfs[10], \
                                                       list_odfs[11], \
                                                       list_odfs[12]

    print('Computing baseline boundaries ...')
    b_baseline, _ = boundary_decoding(decoding_method=method,
                                      obs_i=odf_baseline,
                                      duration_score=syllable_duration,
                                      varin=varin,
                                      threshold=threshold,
                                      hopsize_t=hopsize_t,
                                      viterbiDecoding=viterbiDecoding,
                                      OnsetPeakPickingProcessor=OnsetPeakPickingProcessor)

    print('Computing relu dense boundaries ...')
    b_relu_dense, _ = boundary_decoding(decoding_method=method,
                                        obs_i=odf_relu_dense,
                                        duration_score=syllable_duration,
                                        varin=varin,
                                        threshold=threshold,
                                        hopsize_t=hopsize_t,
                                        viterbiDecoding=viterbiDecoding,
                                        OnsetPeakPickingProcessor=OnsetPeakPickingProcessor)

    print('Computing no dense boundaries ...')
    b_no_dense, _ = boundary_decoding(decoding_method=method,
                                      obs_i=odf_no_dense,
                                      duration_score=syllable_duration,
                                      varin=varin,
                                      threshold=threshold,
                                      hopsize_t=hopsize_t,
                                      viterbiDecoding=viterbiDecoding,
                                      OnsetPeakPickingProcessor=OnsetPeakPickingProcessor)

    print('Computing temporal boundaries ...')
    b_temporal, _ = boundary_decoding(decoding_method=method,
                                      obs_i=odf_temporal,
                                      duration_score=syllable_duration,
                                      varin=varin,
                                      threshold=threshold,
                                      hopsize_t=hopsize_t,
                                      viterbiDecoding=viterbiDecoding,
                                      OnsetPeakPickingProcessor=OnsetPeakPickingProcessor)

    print('Computing bidi lstms 100 boundaries ...')
    b_bidi_lstms_100, _ = boundary_decoding(decoding_method=method,
                                            obs_i=odf_bidi_lstms_100,
                                            duration_score=syllable_duration,
                                            varin=varin,
                                            threshold=threshold,
                                            hopsize_t=hopsize_t,
                                            viterbiDecoding=viterbiDecoding,
                                            OnsetPeakPickingProcessor=OnsetPeakPickingProcessor)

    print('Computing bidi lstms 200 boundaries ...')
    b_bidi_lstms_200, _ = boundary_decoding(decoding_method=method,
                                            obs_i=odf_bidi_lstms_200,
                                            duration_score=syllable_duration,
                                            varin=varin,
                                            threshold=threshold,
                                            hopsize_t=hopsize_t,
                                            viterbiDecoding=viterbiDecoding,
                                            OnsetPeakPickingProcessor=OnsetPeakPickingProcessor)

    print('Computing bidi lstms 400 boundaries ...')
    b_bidi_lstms_400, _ = boundary_decoding(decoding_method=method,
                                            obs_i=odf_bidi_lstms_400,
                                            duration_score=syllable_duration,
                                            varin=varin,
                                            threshold=threshold,
                                            hopsize_t=hopsize_t,
                                            viterbiDecoding=viterbiDecoding,
                                            OnsetPeakPickingProcessor=OnsetPeakPickingProcessor)

    print('Computing 9 layers cnn boundaries ...')
    b_9_layers_cnn, _ = boundary_decoding(decoding_method=method,
                                          obs_i=odf_9_layers_cnn,
                                          duration_score=syllable_duration,
                                          varin=varin,
                                          threshold=threshold,
                                          hopsize_t=hopsize_t,
                                          viterbiDecoding=viterbiDecoding,
                                          OnsetPeakPickingProcessor=OnsetPeakPickingProcessor)

    print('Computing 5 layers cnn boundaries ...')
    b_5_layers_cnn, _ = boundary_decoding(decoding_method=method,
                                          obs_i=odf_5_layers_cnn,
                                          duration_score=syllable_duration,
                                          varin=varin,
                                          threshold=threshold,
                                          hopsize_t=hopsize_t,
                                          viterbiDecoding=viterbiDecoding,
                                          OnsetPeakPickingProcessor=OnsetPeakPickingProcessor)

    print('Computing pretrained boundaries ...')
    b_pretrained, _ = boundary_decoding(decoding_method=method,
                                        obs_i=odf_pretrained,
                                        duration_score=syllable_duration,
                                        varin=varin,
                                        threshold=threshold,
                                        hopsize_t=hopsize_t,
                                        viterbiDecoding=viterbiDecoding,
                                        OnsetPeakPickingProcessor=OnsetPeakPickingProcessor)

    print('Computing retrained boundaries ...')
    b_retrained, _ = boundary_decoding(decoding_method=method,
                                       obs_i=odf_retrained,
                                       duration_score=syllable_duration,
                                       varin=varin,
                                       threshold=threshold,
                                       hopsize_t=hopsize_t,
                                       viterbiDecoding=viterbiDecoding,
                                       OnsetPeakPickingProcessor=OnsetPeakPickingProcessor)

    print('Computing feature extractor a boundaries ...')
    b_feature_extractor_a, _ = boundary_decoding(decoding_method=method,
                                                 obs_i=odf_feature_extractor_a,
                                                 duration_score=syllable_duration,
                                                 varin=varin,
                                                 threshold=threshold,
                                                 hopsize_t=hopsize_t,
                                                 viterbiDecoding=viterbiDecoding,
                                                 OnsetPeakPickingProcessor=OnsetPeakPickingProcessor)

    print('Computing feature extractor b boundaries ...')
    b_feature_extractor_b, _ = boundary_decoding(decoding_method=method,
                                                 obs_i=odf_feature_extractor_b,
                                                 duration_score=syllable_duration,
                                                 varin=varin,
                                                 threshold=threshold,
                                                 hopsize_t=hopsize_t,
                                                 viterbiDecoding=viterbiDecoding,
                                                 OnsetPeakPickingProcessor=OnsetPeakPickingProcessor)

    return b_baseline, b_relu_dense, b_no_dense, b_temporal, b_bidi_lstms_100, b_bidi_lstms_200, \
           b_bidi_lstms_400, b_9_layers_cnn, b_5_layers_cnn, b_pretrained, b_retrained, \
           b_feature_extractor_a, b_feature_extractor_b


b_baseline, b_relu_dense, b_no_dense, b_temporal, b_bidi_lstms_100, b_bidi_lstms_200, \
b_bidi_lstms_400, b_9_layers_cnn, b_5_layers_cnn, b_pretrained, b_retrained, \
b_feature_extractor_a, b_feature_extractor_b = \
    onset_selection(list_odfs=list_odfs,
                    syllable_duration=None,
                    method='peakPicking',
                    threshold=0.20,
                    hopsize_t=hopsize_t,
                    viterbiDecoding=None,
                    OnsetPeakPickingProcessor=OnsetPeakPickingProcessor,
                    varin=varin)

list_boundaries = [b_baseline, b_relu_dense, b_no_dense, b_temporal, b_bidi_lstms_100, b_bidi_lstms_200,
                   b_bidi_lstms_400, b_9_layers_cnn, b_5_layers_cnn, b_pretrained, b_retrained,
                   b_feature_extractor_a, b_feature_extractor_b]

# plot ODFs peak picking, red lines detected boundaries
plot_jingju_odf_colab(log_mel=log_mel,
                      hopsize_t=hopsize_t,
                      list_odfs=list_odfs,
                      groundtruth=boundary_groundtruth,
                      list_boundaries=list_boundaries)

# post processing syllable duration
syllable_duration = np.array([float(sd) for sd in syllable_duration if len(sd)])
# normalize score syllable duration to a sum 1
syllable_duration *= (len(log_mel)*hopsize_t/np.sum(syllable_duration))

b_baseline, b_relu_dense, b_no_dense, b_temporal, b_bidi_lstms_100, b_bidi_lstms_200, \
b_bidi_lstms_400, b_9_layers_cnn, b_5_layers_cnn, b_pretrained, b_retrained, \
b_feature_extractor_a, b_feature_extractor_b = \
    onset_selection(list_odfs=list_odfs,
                    syllable_duration=syllable_duration,
                    method='viterbi',
                    threshold=None,
                    hopsize_t=hopsize_t,
                    viterbiDecoding=viterbiDecoding,
                    OnsetPeakPickingProcessor=None,
                    varin=varin)

list_boundaries = [b_baseline, b_relu_dense, b_no_dense, b_temporal, b_bidi_lstms_100, b_bidi_lstms_200,
                   b_bidi_lstms_400, b_9_layers_cnn, b_5_layers_cnn, b_pretrained, b_retrained,
                   b_feature_extractor_a, b_feature_extractor_b]

# plot ODFs score-informed HMM, red lines detected boundaries
plot_jingju_odf_colab(log_mel=log_mel,
                      hopsize_t=hopsize_t,
                      list_odfs=list_odfs,
                      groundtruth=boundary_groundtruth,
                      list_boundaries=list_boundaries)
