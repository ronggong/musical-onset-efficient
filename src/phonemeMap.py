#!/usr/bin/env python
# -*- coding: utf-8 -*-

pho_set_all = [u'', u'U^', u'an', u'in', u' ', u'aI^', u'7N', u'yn', u'@n', u'eI^', u'1', u'7', u'iN', u'9',
               u"r\\'", u'?', u'@', u'E', u'H', u'M', u'O', u'N', u'oU^', u'AN',
               u'AU^', u'a', u'c', u'En', u'f', u'i', u'k', u'j', u'm', u'l', u'o', u'n', u'UN',
               u'u', u'w', u'y', u'x']

pho_set_initials = [u"r\\'",
                       u'k',
                       u'f',
                       u'x',
                       u'm',
                       u'l',
                       u'n',
                       u'w',
                       u'j',
                       u'N',

                       u'c']

pho_set_finals = [u'', u'U^', u'an', u'in', u'aI^', u'7N', u'yn', u'@n', u'eI^', u'1', u'7', u'iN', u'9',
               u"r\\'", u'?', u'@', u'E', u'H', u'M', u'O', u'N', u'oU^', u'AN',
               u'AU^', u'a', u'En', u'i', u'j', u'o',u'n', u'UN',
               u'u', u'w', u'y', u'x']

dic_pho_map = {u'': u'sil',
               u'?': u'?',
               u'an': u'an',
               u'in': u'in',
               u'aI^': u'aI^',
               u'7N': u'SN',
               u'yn': u'in', # reason: no much yn samples
               u'@n': u'@n',
               u'eI^': u'eI^',
               u'1': u'ONE',
               u'7': u'S',
               u'iN': u'iNiN',
               u'9': u'S',  # reason: no much 9 samples
               u'@':u'@',
               u'E':u'E',
               u'H':u'y',  # reason: no much H samples
               u'M':u'MM',
               u'O':u'O',
               u'N':u'N',
               u'oU^':u'oU^',
               u'AN':u'ANAN',
               u'AU^':u'AU^',
               u'a':u'a',
               u'En':u'EnEn',
               u'i':u'i',
               u'UN':u'UN',
               u'u':u'u',
               u'w':u'u',
               u'U^': u'u',
               u'y':u'y',
               u'j':u'j',

               u"r\\'": u'rr',

               u'm':u'vc',
               u'l':u'vc',
               u'n':u'vc',

               u'c':u'nvc',
               u'f':u'nvc',
               u'k':u'nvc',
               u's':u'nvc',
               u'x':u'nvc'}

dic_pho_label = {u'sil':0,
               u'?':1,
               u'an':2,
               u'in':3,
               u'aI^':4,
               u'SN':5,
               #u'yn':6,
               u'@n':6,
               u'eI^':7,
               u'ONE':8,
               u'S':9,
               u'iNiN':10,
               #u'9':11,
               u'@':11,
               u'E':12,
               #u'H':15,
               u'MM':13,
               u'O':14,
               u'N':15,
               u'oU^':16,
               u'ANAN':17,
               u'AU^':18,
               u'a':19,
               u'EnEn':20,
               u'i':21,
               u'UN':22,
               u'u':23,
               u'y':24,
               u'j':25,
               u'rr':26,
               u'vc':27,
               u'nvc':28}

dic_pho_label_inv = {0: u'sil',
                     1: u'?',
                     2: u'an',
                     3: u'in',
                     4: u'aI^',
                     5: u'SN',
                     6: u'@n',
                     7: u'eI^',
                     8: u'ONE',
                     9: u'S',
                     10: u'iNiN',
                     11: u'@',
                     12: u'E',
                     13: u'MM',
                     14: u'O',
                     15: u'N',
                     16: u'oU^',
                     17: u'ANAN',
                     18: u'AU^',
                     19: u'a',
                     20: u'EnEn',
                     21: u'i',
                     22: u'UN',
                     23: u'u',
                     24: u'y',
                     25: u'j',
                     26: u'rr',
                     27: u'vc',
                     28: u'nvc'}

dic_pho_map_topo = {u'': u'sil',
               u'?': u'?',
               u'an': u'an',
               u'in': u'in',
               u'aI^': u'aI^',
               u'7N': u'SN',
               u'yn': u'yn',
               u'@n': u'@n',
               u'eI^': u'eI^',
               u'1': u'1',
               u'7': u'7',
               u'iN': u'iN',
               u'9': u'9',
               u'@':u'@',
               u'E':u'E',
               u'H':u'H',
               u'M':u'M',
               u'O':u'O',
               u'N':u'N',
               u'oU^':u'oU^',
               u'AN':u'AN',
               u'AU^':u'AU^',
               u'a':u'a',
               u'En':u'En',
               u'i':u'i',
               u'UN':u'UN',
               u'u':u'u',
               u'w':u'u',
               u'U^': u'u',
               u'y':u'y',
               u'j':u'j',

               u"r\\'": u'r',

               u'm':u'vc',
               u'l':u'vc',
               u'n':u'vc',

               u'c':u'nvc',
               u'f':u'nvc',
               u'k':u'nvc',
               u's':u'nvc',
               u'x':u'nvc'}

tails_comb_n        = [u'an_n', u'in_n',u'i_n', u'@_n', u'_n', u'@n_n', u'a_n', u'yn_n',u'y_n', u'E_n', u'En_n']
tails_comb_N        = [u'AN_N', u'iN_N', u'UN_N', u'7N_N', u'_N', u'a_N',  u'AU^_N',u'oU^_N']
tails_comb_i        = [u'u_i', u'O_i', u'aI^_i', u'E_i', u'eI^_i',u'7_i']
tails_comb_u        = [u'oU^_u', u'AU^_u',  u'UN_u']

tails_comb_n_map        = [u'an_vc', u'in_vc',u'i_vc', u'@_vc', u'sil_vc', u'@n_vc', u'a_vc', u'yn_vc',u'y_vc', u'E_vc', u'EnEn_vc']
tails_comb_N_map        = [u'ANAN_N', u'iNiN_N', u'UN_N', u'SN_N', u'sil_N', u'a_N', u'AU^_N',u'oU^_N']
tails_comb_i_map        = [u'u_i', u'O_i', u'aI^_i', u'E_i', u'eI^_i',u'S_i']
tails_comb_u_map        = [u'oU^_u', u'AU^_u', u'UN_u']

# phoneme categories
vowels              = [u'1',u'7',u'9',u'@',u'E',u'H',u'M',u'O',u'a',u'i',u'u',u'U^',u'y']
semivowels          = [u'w',u'j']
diphtongs           = [u'aI^',u'eI^',u'oU^',u'AU^']
compoundfinals      = [u'an',u'in',u'7N',u'yn',u'@n',u'iN',u'AN',u'En',u'UN']
nonvoicedconsonants = [u'c',u'f',u'k',u's',u'x']
voicedconsonants    = [u'N',u"r\\'",u'm',u'l',u'n']
silornament         = [u'',u'?']

types_phoneme       = ['vowels','semivowels','diphtongs','compoundfinals','nonvoicedconsonants','voicedconsonants','silornament']

trans_phoneme       = ['vowels_vowels', 'vowels_semivowels', 'vowels_diphtongs', 'vowels_compoundfinals', 'vowels_nonvoicedconsonants', 'vowels_voicedconsonants', 'vowels_silornament', 'semivowels_vowels', 'semivowels_semivowels', 'semivowels_diphtongs', 'semivowels_compoundfinals', 'semivowels_nonvoicedconsonants', 'semivowels_voicedconsonants', 'semivowels_silornament', 'diphtongs_vowels', 'diphtongs_semivowels', 'diphtongs_diphtongs', 'diphtongs_compoundfinals', 'diphtongs_nonvoicedconsonants', 'diphtongs_voicedconsonants', 'diphtongs_silornament', 'compoundfinals_vowels', 'compoundfinals_semivowels', 'compoundfinals_diphtongs', 'compoundfinals_compoundfinals', 'compoundfinals_nonvoicedconsonants', 'compoundfinals_voicedconsonants', 'compoundfinals_silornament', 'nonvoicedconsonants_vowels', 'nonvoicedconsonants_semivowels', 'nonvoicedconsonants_diphtongs', 'nonvoicedconsonants_compoundfinals', 'nonvoicedconsonants_nonvoicedconsonants', 'nonvoicedconsonants_voicedconsonants', 'nonvoicedconsonants_silornament', 'voicedconsonants_vowels', 'voicedconsonants_semivowels', 'voicedconsonants_diphtongs', 'voicedconsonants_compoundfinals', 'voicedconsonants_nonvoicedconsonants', 'voicedconsonants_voicedconsonants', 'voicedconsonants_silornament', 'silornament_vowels', 'silornament_semivowels', 'silornament_diphtongs', 'silornament_compoundfinals', 'silornament_nonvoicedconsonants', 'silornament_voicedconsonants', 'silornament_silornament']