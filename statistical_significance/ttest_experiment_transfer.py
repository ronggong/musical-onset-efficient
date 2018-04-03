from scipy.stats import ttest_ind
import pickle
import os

"""
Jan less deep
Jan pretrained
Jan weight initialization
Jan feature extraction
Jan deep feature extraction
"""

# # artist_filter peak picking
recall_precison_schluter_less_deep_pp = pickle.load(open(os.path.join('./data/bock/simpleWeighting', '5_layers_cnn.pkl'), 'r'))
recall_precison_schluter_pretrained_pp = pickle.load(open(os.path.join('./data/bock/simpleWeighting', 'pretrained.pkl'), 'r'))
recall_precison_schluter_weight_init_pp = pickle.load(open(os.path.join('./data/bock/simpleWeighting', 'retrained.pkl'), 'r'))
recall_precison_schluter_feature_extraction_pp = pickle.load(open(os.path.join('./data/bock/simpleWeighting', 'feature_extractor_a.pkl'), 'r'))
recall_precison_schluter_deep_feature_extraction_pp = pickle.load(open(os.path.join('./data/bock/simpleWeighting', 'feature_extractor_b.pkl'), 'r'))

f1_schluter_less_deep_pp = [rpf[2] for rpf in recall_precison_schluter_less_deep_pp]
f1_schluter_pretrained_pp = [rpf[2] for rpf in recall_precison_schluter_pretrained_pp]
f1_schluter_weight_init_pp = [rpf[2] for rpf in recall_precison_schluter_weight_init_pp]
f1_schluter_feature_extraction_pp = [rpf[2] for rpf in recall_precison_schluter_feature_extraction_pp]
f1_schluter_deep_feature_extraction_pp = [rpf[2] for rpf in recall_precison_schluter_deep_feature_extraction_pp]

# # artist_filter peak picking
f1_artist_filter_less_deep_pp = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', '5_layers_cnn_peakPickingMadmom.pkl'), 'r'))
f1_artist_filter_pretrained_pp = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'pretrained_peakPickingMadmom.pkl'), 'r'))
f1_artist_filter_weight_init_pp = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'retrained_peakPickingMadmom.pkl'), 'r'))
f1_artist_filter_feature_extraction_pp = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'feature_extractor_a_peakPickingMadmom.pkl'), 'r'))
f1_artist_filter_deep_feature_extraction_pp = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'feature_extractor_b_peakPickingMadmom.pkl'), 'r'))

# # artist_filter viterbi no label
f1_artist_filter_less_deep_nl = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', '5_layers_cnn_viterbi_nolabel.pkl'), 'r'))
f1_artist_filter_pretrained_nl = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'pretrained_viterbi_nolabel.pkl'), 'r'))
f1_artist_filter_weight_init_nl = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'retrained_viterbi_nolabel.pkl'), 'r'))
f1_artist_filter_feature_extraction_nl = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'feature_extractor_a_viterbi_nolabel.pkl'), 'r'))
f1_artist_filter_deep_feature_extraction_nl = pickle.load(open(os.path.join('./data/jingju/simpleWeighting', 'feature_extractor_b_viterbi_nolabel.pkl'), 'r'))


def pValueAll(f1_jan,
              f1_pretrained,
              f1_weight_init,
              f1_feature_extraction,
              f1_deep_feature_extraction):

    _, p_jan_pretrained = ttest_ind(f1_jan, f1_pretrained, equal_var=False)

    _, p_jan_weight_init = ttest_ind(f1_jan, f1_weight_init, equal_var=False)

    _, p_jan_feature_extraction = ttest_ind(f1_jan, f1_feature_extraction, equal_var=False)

    _, p_jan_deep_feature_extraction = ttest_ind(f1_jan, f1_deep_feature_extraction, equal_var=False)

    print(p_jan_pretrained)
    print(p_jan_weight_init)
    print(p_jan_feature_extraction)
    print(p_jan_deep_feature_extraction)

    return p_jan_pretrained, p_jan_weight_init, p_jan_feature_extraction, p_jan_deep_feature_extraction


def stat_schluter_pp():
    pValueAll(f1_schluter_less_deep_pp,
              f1_schluter_pretrained_pp,
              f1_schluter_weight_init_pp,
              f1_schluter_feature_extraction_pp,
              f1_schluter_deep_feature_extraction_pp)

def stat_artist_pp():
    pValueAll(f1_artist_filter_less_deep_pp,
              f1_artist_filter_pretrained_pp,
              f1_artist_filter_weight_init_pp,
              f1_artist_filter_feature_extraction_pp,
              f1_artist_filter_deep_feature_extraction_pp)


def stat_artist_nl():
    pValueAll(f1_artist_filter_less_deep_nl,
              f1_artist_filter_pretrained_nl,
              f1_artist_filter_weight_init_nl,
              f1_artist_filter_feature_extraction_nl,
              f1_artist_filter_deep_feature_extraction_nl)

if __name__ == '__main__':
    print('peak picking jingju:')
    stat_artist_pp()
    print('score-informed HMM jingju')
    stat_artist_nl()
    print('peak picking Bock')
    stat_schluter_pp()