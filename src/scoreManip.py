from pinyinMap import dic_pinyin_2_initial_final_map, dic_initial_2_sampa, dic_final_2_sampa
from phonemeMap import nonvoicedconsonants, dic_pho_map
import numpy as np
import os
import json

currentPath = os.path.dirname(__file__)
parentPath  = os.path.join(currentPath, '..')
with open(os.path.join(parentPath, 'trainingData', 'dict_centroid_dur.json'), 'r') as openfile:
    dict_centroid_dur = json.load(openfile)

def parsePinyin(pinyin):
    """
    pinyin to sampa, omit nonvoiced consonants
    :param pinyin:
    :return:
    """
    dict_pinyin = dic_pinyin_2_initial_final_map[pinyin]
    initial     = dict_pinyin['initial']
    final       = dict_pinyin['final']
    initial_xsampa  = dic_initial_2_sampa[initial]
    final_xsampa    = dic_final_2_sampa[final]

    if initial_xsampa in nonvoicedconsonants or not len(initial_xsampa):
        xsample_syllable =  final_xsampa
    else:
        xsample_syllable =  [initial_xsampa]+final_xsampa

    return [dic_pho_map[xs] for xs in xsample_syllable]


def phonemeDurationForSyllable(xsampa_syllable, dur_syllable):
    """
    phoneme durations for syllable which has dur_syllable
    :param xsampa_syllable:
    :param dur_syllable:
    :return:
    """
    xsampa_pho_centroid_dur = np.array([dict_centroid_dur[xs] for xs in xsampa_syllable])
    return (xsampa_pho_centroid_dur/np.sum(xsampa_pho_centroid_dur))*dur_syllable


def phonemeDurationForLine(pinyin_score, dur_score):
    """
    parse xsampa and duration of a line
    :param pinyin_score:
    :param dur_score:
    :return:
    """
    xsampa_line = []
    phonemeDurationLine = []
    idx_syllable_start_state = [0]
    for ii in range(len(pinyin_score)):
        pinyin          = pinyin_score[ii]
        dur_syllable    = dur_score[ii]
        xsampa_syllable = parsePinyin(pinyin)
        # print(pinyin, xsampa_syllable)
        # print(xsampa_syllable, dur_syllable)
        phonemeDurationSyllable = phonemeDurationForSyllable(xsampa_syllable,dur_syllable)
        idx_syllable_start_state.append(idx_syllable_start_state[-1]+len(xsampa_syllable))
        xsampa_line += xsampa_syllable
        phonemeDurationLine += phonemeDurationSyllable.tolist()
    idx_syllable_start_state = idx_syllable_start_state[:-1]

    return xsampa_line, phonemeDurationLine, idx_syllable_start_state