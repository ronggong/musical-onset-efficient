import os
import sys
import gzip
import pickle
import cPickle
import numpy as np
from keras.models import load_model
from experiment_process_helper import odf_calculation_crnn

sys.path.append(os.path.join(os.path.dirname(__file__), "../src/"))

from audio_preprocessing import getMFCCBands2DMadmom
from utilFunctions import featureReshape
from utilFunctions import smooth_obs
from training_scripts.models_CRNN import jan_original
from plot_code import plot_jingju_odf_colab

hopsize_t = 0.01
fs = 44100


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


path_models_jingju = './pretrained_models/jingju/simpleWeighting'
path_models_bock = './pretrained_models/bock/simpleWeighting'

wav_jingju = './inputs/audio_5.wav'

path_scaler_no_rnn = os.path.join(path_models_jingju, 'scaler_jan_no_rnn.pkl')
scaler_no_rnn = pickle.load(open(path_scaler_no_rnn))

path_scaler_crnn = os.path.join(path_models_jingju, 'scaler_jingju_crnn_phrase.pkl')
scaler_crnn = pickle.load(open(path_scaler_crnn))

path_scaler_bock = os.path.join(path_models_bock, 'scaler_bock_0.pickle.gz')
scaler_bock = cPickle.load(gzip.open(path_scaler_bock))

# load score
nested_list, syllable_duration = pickle.load(open('./inputs/input_jingju_5.pkl', 'r'))

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

# plot
plot_jingju_odf_colab(log_mel=log_mel,
                      hopsize_t=hopsize_t,
                      odf_baseline=odf_baseline,
                      odf_relu_dense=odf_relu_dense,
                      odf_no_dense=odf_no_dense,
                      odf_temporal=odf_temporal,
                      odf_bidi_lstms_100=odf_bidi_lstms_100,
                      odf_bidi_lstms_200=odf_bidi_lstms_200,
                      odf_bidi_lstms_400=odf_bidi_lstms_400,
                      odf_9_layers_cnn=odf_9_layers_cnn,
                      odf_5_layers_cnn=odf_5_layers_cnn,
                      odf_pretrained=odf_pretrained,
                      odf_retrained=odf_retrained,
                      odf_feature_extractor_a=odf_feature_extractor_a,
                      odf_feature_extractor_b=odf_feature_extractor_b)
