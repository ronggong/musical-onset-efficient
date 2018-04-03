from scipy.stats import ttest_ind
import pickle
import os

"""
Jan relu dense
Jan no dense
temporal sigmoid dense
Jan deep no dense (much slow than jan sigmoid or relu)
Jan less deep
Jan (Bidirectional LSTM 400 length, 93.5 percentile)
Jan (Bidi LSTM 200 length 85 percentile)
Jan (Bidi LSTM 100 length 66.5 percentile)
"""

# bock
recall_precision_f1_jan = pickle.load(open(os.path.join('./data/bock/simpleWeighting', 'baseline.pkl'), 'r'))

recall_precision_f1_no_dense = pickle.load(open(os.path.join('./data/bock/simpleWeighting', 'no_dense.pkl'), 'r'))
recall_precision_f1_relu = pickle.load(open(os.path.join('./data/bock/simpleWeighting', 'relu_dense.pkl'), 'r'))
recall_precision_f1_temporal = pickle.load(open(os.path.join('./data/bock/simpleWeighting', 'temporal.pkl'), 'r'))
recall_precision_f1_deep = pickle.load(open(os.path.join('./data/bock/simpleWeighting', '9_layers_cnn.pkl'), 'r'))
recall_precision_f1_less_deep = pickle.load(open(os.path.join('./data/bock/simpleWeighting', '5_layers_cnn.pkl'), 'r'))
recall_precision_f1_bidi_400 = pickle.load(open(os.path.join('./data/bock/simpleWeighting', 'bidi_lstms_400.pkl'), 'r'))
recall_precision_f1_bidi_200 = pickle.load(open(os.path.join('./data/bock/simpleWeighting', 'bidi_lstms_200.pkl'), 'r'))
recall_precision_f1_bidi_100 = pickle.load(open(os.path.join('./data/bock/simpleWeighting', 'bidi_lstms_100.pkl'), 'r'))

f1_schluter_jan = [rpf[2] for rpf in recall_precision_f1_jan]

f1_schluter_no_dense = [rpf[2] for rpf in recall_precision_f1_no_dense]
f1_schluter_relu = [rpf[2] for rpf in recall_precision_f1_relu]
f1_schluter_temporal = [rpf[2] for rpf in recall_precision_f1_temporal]
f1_schluter_deep = [rpf[2] for rpf in recall_precision_f1_deep]
f1_schluter_less_deep = [rpf[2] for rpf in recall_precision_f1_less_deep]
f1_schluter_bidi_400 = [rpf[2] for rpf in recall_precision_f1_bidi_400]
f1_schluter_bidi_200 = [rpf[2] for rpf in recall_precision_f1_bidi_200]
f1_schluter_bidi_100 = [rpf[2] for rpf in recall_precision_f1_bidi_100]


# # artist_filter peak picking
f1_artist_filter_jan_pp = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'baseline_peakPickingMadmom.pkl'), 'r'))

f1_artist_filter_no_dense_pp = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'no_dense_peakPickingMadmom.pkl'), 'r'))
f1_artist_filter_relu_pp = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'relu_dense_peakPickingMadmom.pkl'), 'r'))
f1_artist_filter_temporal_pp = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'jordi_temporal_artist_filter_madmom_early_stopping_more_params_peakPickingMadmom.pkl'), 'r'))
f1_artist_filter_deep_pp = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', '9_layers_cnn_peakPickingMadmom.pkl'), 'r'))
f1_artist_filter_less_deep_pp = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', '5_layers_cnn_peakPickingMadmom.pkl'), 'r'))
f1_artist_filter_bidi_400_pp = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'bidi_lstms_400_peakPickingMadmom.pkl'), 'r'))
f1_artist_filter_bidi_200_pp = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'bidi_lstms_200_peakPickingMadmom.pkl'), 'r'))
f1_artist_filter_bidi_100_pp = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'bidi_lstms_100_peakPickingMadmom.pkl'), 'r'))

# # artist_filter viterbi no label
f1_artist_filter_jan_nl = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'baseline_viterbi_nolabel.pkl'), 'r'))

f1_artist_filter_no_dense_nl = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'no_dense_viterbi_nolabel.pkl'), 'r'))
f1_artist_filter_relu_nl = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'relu_dense_viterbi_nolabel.pkl'), 'r'))
f1_artist_filter_temporal_nl = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'jordi_temporal_artist_filter_madmom_early_stopping_more_params_viterbi_nolabel.pkl'), 'r'))
f1_artist_filter_deep_nl = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', '9_layers_cnn_viterbi_nolabel.pkl'), 'r'))
f1_artist_filter_less_deep_nl = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', '5_layers_cnn_viterbi_nolabel.pkl'), 'r'))
f1_artist_filter_bidi_400_nl = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'bidi_lstms_400_viterbi_nolabel.pkl'), 'r'))
f1_artist_filter_bidi_200_nl = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'bidi_lstms_200_viterbi_nolabel.pkl'), 'r'))
f1_artist_filter_bidi_100_nl = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'bidi_lstms_100_viterbi_nolabel.pkl'), 'r'))


def pValueAll(f1_jan, f1_no_dense, f1_relu, f1_temporal, f1_deep, f1_less_deep, f1_bidi_400, f1_bidi_200, f1_bidi_100):

    _, p_jan_no_dense = ttest_ind(f1_jan, f1_no_dense, equal_var=False)

    _, p_jan_relu = ttest_ind(f1_jan, f1_relu, equal_var=False)

    _, p_jan_temporal = ttest_ind(f1_jan, f1_temporal, equal_var=False)

    _, p_jan_deep = ttest_ind(f1_jan, f1_deep, equal_var=False)

    _, p_jan_less_deep = ttest_ind(f1_jan, f1_less_deep, equal_var=False)

    _, p_jan_bidi_400 = ttest_ind(f1_jan, f1_bidi_400, equal_var=False)

    _, p_jan_bidi_200 = ttest_ind(f1_jan, f1_bidi_200, equal_var=False)

    _, p_jan_bidi_100 = ttest_ind(f1_jan, f1_bidi_100, equal_var=False)


    return p_jan_no_dense, p_jan_relu, p_jan_temporal, p_jan_deep, \
           p_jan_less_deep, p_jan_bidi_400, p_jan_bidi_200, p_jan_bidi_100


def stat_schluter():
    p_jan_no_dense, p_jan_relu, p_jan_temporal, p_jan_deep, \
    p_jan_less_deep, p_jan_bidi_400, p_jan_bidi_200, p_jan_bidi_100 = \
        pValueAll(f1_jan=f1_schluter_jan,
              f1_no_dense=f1_schluter_no_dense,
              f1_relu=f1_schluter_relu,
              f1_temporal=f1_schluter_temporal,
              f1_deep=f1_schluter_deep,
              f1_less_deep=f1_schluter_less_deep,
              f1_bidi_400=f1_schluter_bidi_400,
              f1_bidi_200=f1_schluter_bidi_200,
              f1_bidi_100=f1_schluter_bidi_100)

    print p_jan_no_dense
    print p_jan_relu
    print p_jan_temporal
    print p_jan_deep
    print p_jan_less_deep
    print p_jan_bidi_400
    print p_jan_bidi_200
    print p_jan_bidi_100


def stat_artist_pp():
    p_jan_no_dense, p_jan_relu, p_jan_temporal, p_jan_deep, \
    p_jan_less_deep, p_jan_bidi_400, p_jan_bidi_200, p_jan_bidi_100 = \
        pValueAll(f1_jan=f1_artist_filter_jan_pp,
                  f1_no_dense=f1_artist_filter_no_dense_pp,
                  f1_relu=f1_artist_filter_relu_pp,
                  f1_temporal=f1_artist_filter_temporal_pp,
                  f1_deep=f1_artist_filter_deep_pp,
                  f1_less_deep=f1_artist_filter_less_deep_pp,
                  f1_bidi_400=f1_artist_filter_bidi_400_pp,
                  f1_bidi_200=f1_artist_filter_bidi_200_pp,
                  f1_bidi_100=f1_artist_filter_bidi_100_pp)

    print p_jan_no_dense
    print p_jan_relu
    print p_jan_temporal
    print p_jan_deep
    print p_jan_less_deep
    print p_jan_bidi_400
    print p_jan_bidi_200
    print p_jan_bidi_100


def stat_artist_nl():
    p_jan_no_dense, p_jan_relu, p_jan_temporal, p_jan_deep, \
    p_jan_less_deep, p_jan_bidi_400, p_jan_bidi_200, p_jan_bidi_100 = \
        pValueAll(f1_jan=f1_artist_filter_jan_nl,
                  f1_no_dense=f1_artist_filter_no_dense_nl,
                  f1_relu=f1_artist_filter_relu_nl,
                  f1_temporal=f1_artist_filter_temporal_nl,
                  f1_deep=f1_artist_filter_deep_nl,
                  f1_less_deep=f1_artist_filter_less_deep_nl,
                  f1_bidi_400=f1_artist_filter_bidi_400_nl,
                  f1_bidi_200=f1_artist_filter_bidi_200_nl,
                  f1_bidi_100=f1_artist_filter_bidi_100_nl)

    print p_jan_no_dense
    print p_jan_relu
    print p_jan_temporal
    print p_jan_deep
    print p_jan_less_deep
    print p_jan_bidi_400
    print p_jan_bidi_200
    print p_jan_bidi_100

if __name__ == '__main__':
    print('peak picking jingju:')
    stat_artist_pp()
    print('score-informed HMM jingju')
    stat_artist_nl()
    print('peak picking Bock')
    stat_schluter()