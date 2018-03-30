import os
from itertools import combinations
from operator import itemgetter
from file_path_jingju_no_rnn import *
import numpy as np
import textgridParser
import scoreParser
from labParser import lab2WordList


def testRecordings(boundaries,proportion_testset):
    '''
    find the test recording numbers which meets the proportion
    :param boundaries: a list of boundary number of each recording
    :param proportion_testset:
    :return: a list of test recordings
    '''

    sum_boundaries = sum(boundaries)
    boundaries     = np.array(boundaries)
    subsets        = []

    for ii in range(1,len(boundaries)):
        # print(ii, len(boundaries))
        for subset in combinations(range(len(boundaries)),ii):
            subsets.append([subset,abs(sum(boundaries[np.array(subset)])/float(sum_boundaries)-proportion_testset)])

    subsets        = np.array(subsets)
    subsets_sorted = subsets[np.argsort(subsets[:,1]),0]

    return subsets_sorted[0]

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


def getBoundaryNumber(textgrid_path, score_path):
    """
    output a list to show the syllable number for each aria,
    the syllable number is extracted from the textgrid
    the textgrid needs to have a score
    :param textgrid_path:
    :param score_path:
    :return:
    """
    listOnset = []
    list_file_path_name = []
    for file_path_name in os.walk(textgrid_path):
        list_file_path_name.append(file_path_name)

    list_artist_level_path = list_file_path_name[0][1]

    for artist_path in list_artist_level_path:

        textgrid_artist_path = join(textgrid_path, artist_path)
        recording_names = [f for f in os.listdir(textgrid_artist_path) if os.path.isfile(join(textgrid_artist_path, f))]

        for rn in recording_names:
            rn = rn.split('.')[0]
            groundtruth_textgrid_file = join(textgrid_path, artist_path, rn+'.TextGrid')
            # if artist_path=='danAll' or artist_path=='laosheng':
            #     score_file = join(score_path, rn+'.csv')
            # else:
            score_file = join(score_path, artist_path, rn + '.csv')

            if not os.path.isfile(score_file):
                continue
            # print(groundtruth_textgrid_file)

            lineList = textgridParser.textGrid2WordList(groundtruth_textgrid_file, whichTier='line')
            utteranceList = textgridParser.textGrid2WordList(groundtruth_textgrid_file, whichTier='dianSilence')

            # parse lines of groundtruth
            nestedUtteranceLists, numLines, numUtterances = textgridParser.wordListsParseByLines(lineList,
                                                                                                 utteranceList)

            # parse score
            _, utterance_durations, bpm = scoreParser.csvDurationScoreParser(score_file)

            # create the ground truth lab files
            numOnset = 0
            for idx, list in enumerate(nestedUtteranceLists):
                try:
                    if float(bpm[idx]):
                        print('Counting onset number ... ' + rn + ' phrase ' + str(idx + 1))

                        ul = list[1]
                        numOnsetLine = len(ul) - 1  # we don't count the first onset
                        numOnset += numOnsetLine
                except IndexError:
                    print(idx, 'not exist for recording', rn)

            listOnset += [[artist_path, rn, numOnset]]

    return listOnset

def getBoundaryNumberLab(groundtruthLab_path, score_path):
    """
    output a list to show the syllable number for each aria,
    the syllable number is extracted from the groundtruth lab
    the textgrid needs to have a score
    :param groundtruthLab_path:
    :param score_path:
    :return:
    """

    listOnset = []

    recording_names = [f for f in os.listdir(groundtruthLab_path) if os.path.isfile(join(groundtruthLab_path, f))]
    for rn in recording_names:
        rn = rn.split('.')[0]
        groundtruth_lab_file = join(groundtruthLab_path, rn + '.lab')
        score_file = join(score_path, rn + '.csv')
        if not os.path.isfile(score_file):
            continue

        list_syllable = lab2WordList(groundtruth_lab_file, label=False)

        numOnset = 0
        for idx, list in enumerate(list_syllable):
            try:
                print('Counting onset number ... ' + rn + ' phrase ' + str(idx + 1))

                numOnsetLine = len(list) - 1
                numOnset += numOnsetLine
            except IndexError:
                print(idx, 'not exist for recording', rn)
        listOnset += [['dummy_artist_path', rn, numOnset]]
    return listOnset

def getTestTrainrecordingsRiyaz():
    # list_onset_riyaz = getBoundaryNumberLab(groundtruthLab_path=riyaz_groundtruthlab_path, score_path=riyaz_score_path)
    # numOnset = [n[2] for n in list_onset_riyaz]
    # recording_names = [n[1] for n in list_onset_riyaz]
    # numOnset0 = numOnset[:15]
    # numOnset1 = numOnset[15:30]
    # numOnset2 = numOnset[30:45]
    # numOnset3 = numOnset[45:]
    # print('riyaz 0 test number', testRecordings(numOnset0, 0.2))  # (0, 7, 8, 11, 12, 13, 14)
    # print('riyaz 1 test number', testRecordings(numOnset1, 0.2))  # (0, 3, 6, 9, 10, 13)
    # print('riyaz 2 test number', testRecordings(numOnset2, 0.2))  # (1, 6, 7, 10)
    # print('riyaz 3 test number', testRecordings(numOnset3, 0.2))  # (0, 4, 9)
    #
    # print(sum(itemgetter(*[0, 7, 8, 11, 12, 13, 14, 15, 18, 21, 24, 25, 28, 31, 36, 37, 40, 45, 49, 54])(numOnset))/float(sum(numOnset)))
    #
    # recordingsTestRiyaz = [['', list_onset_riyaz[ii][1]] for ii in
    #                               [0, 7, 8, 11, 12, 13, 14, 15, 18, 21, 24, 25, 28, 31, 36, 37, 40, 45, 49, 54]]
    # recordingsTrainRiyaz = [['', list_onset_riyaz[ii][1]] for ii in
    #                                range(len(list_onset_riyaz)) if ii not in [0, 7, 8, 11, 12, 13, 14, 15, 18, 21, 24, 25, 28, 31, 36, 37, 40, 45, 49, 54]]

    recordingsTestRiyaz = [['', 'SSazJAOgvJ'], ['', 'WUfMbICU2t'], ['', 'Rmh6nTRNRV'], ['', 'xNgxgNL16M'], ['', 'cng5OqLTFD'],
     ['', 'lo0uEZ1XnZ'], ['', 'fzKAFbSQpa'], ['', 'mMMVlKPVg3'], ['', 'AENiVWRkhl'], ['', 'rwoAN7KATB'],
     ['', 'PNrMzI3mL4'], ['', 'uHPUyd16GD'], ['', '0C4LcxSbt1'], ['', '0DcjUiF456'], ['', '0QZth6LDtw'],
     ['', 'J7K6T9YZsF'], ['', 'Gl5okMvDzr'], ['', 'KjsJQc0lOW'], ['', '0dXVbCCxcA'], ['', 'HKswgkFp6C']]

    recordingsTrainRiyaz = [['', '0A3SPhtRMH'], ['', '6aazlC7DGc'], ['', '0qKAEL5BtR'], ['', '1x0pmfE8Rq'], ['', 'rFFopZAGb0'],
     ['', 'm75IOPxPIP'], ['', 'GXYiJc3qbE'], ['', '0GmiArFSAW'], ['', 'QB7rlJtYsE'], ['', '0PdkLJ9QAy'],
     ['', 'skGpjqnmso'], ['', 'V2SW0k0WsK'], ['', '0dFpOMlA9p'], ['', '01jroGVdYr'], ['', 'a0oYZhwo04'],
     ['', 'w4A85oN25D'], ['', 'MhRkNjUAvf'], ['', 'WvryUiwpwP'], ['', '00PdvbkGvD'], ['', 'CJsHSlSBbZ'],
     ['', '0FP3Qks8xh'], ['', 'kaM2vR1SLz'], ['', 'Q4l0OGla6j'], ['', 'c5Ro2tqfiA'], ['', 'X408AFbLUM'],
     ['', '0eHOPcBTn4'], ['', 'hxiKs7h9IT'], ['', 'ZJYl5507ky'], ['', 'FLgffF2z89'], ['', '8P2aWWEeZd'],
     ['', 'PFHQiuzgEC'], ['', '6DvkHWOdKQ'], ['', '0t4OAPATPi'], ['', 'aNjpLW1PdF'], ['', 'ttcuWsnYHF'],
     ['', 'Nnh9YzFn19'], ['', 'S5jhIGJ15A'], ['', '1VgNgElHhO'], ['', 'KmN1QDEPKH']]

    return recordingsTestRiyaz, recordingsTrainRiyaz


def getTestTrainRecordingsMaleFemale():
    """
    only split male female, new split
    :return:
    """
    list_onset_nacta2017 = getBoundaryNumber(textgrid_path=nacta2017_textgrid_path, score_path=nacta2017_score_path)
    list_onset_nacta = getBoundaryNumber(textgrid_path=nacta_textgrid_path, score_path=nacta_score_path)

    list_onset_nacta.remove(['laosheng', 'lsxp-Guo_liao_yi-Wen_zhao_guan01-upf', 18])
    list_onset_nacta.remove(['laosheng', 'lsxp-Guo_liao_yi-Wen_zhao_guan02-qm', 19])

    listOnsetNacta2017Male = [n for n in list_onset_nacta2017 if '2017' in n[0] and 'ls' == n[1][:2]]
    listOnsetNacta2017Fem = [n for n in list_onset_nacta2017 if '2017' in n[0] and 'da' == n[1][:2]]
    numOnsetNacta2017Male = [n[2] for n in list_onset_nacta2017 if '2017' in n[0] and 'ls'==n[1][:2]]
    numOnsetNacta2017Fem = [n[2] for n in list_onset_nacta2017 if '2017' in n[0] and 'da'==n[1][:2]]

    listOnsetNactaMale = [n for n in list_onset_nacta if '2017' not in n[0] and 'ls' == n[1][:2]]
    listOnsetNactaFem = [n for n in list_onset_nacta if '2017' not in n[0] and 'da' == n[1][:2]]
    numOnsetNactaMale = [n[2] for n in list_onset_nacta if '2017' not in n[0] and 'ls'==n[1][:2]]
    numOnsetNactaFem = [n[2] for n in list_onset_nacta if '2017' not in n[0] and 'da'==n[1][:2]]

    print(sum(numOnsetNacta2017Male), sum(numOnsetNacta2017Fem), sum(numOnsetNactaMale), sum(numOnsetNactaFem))

    # segment the onset number list to accelerate the combination calculation
    numOnsetNacta2017Male0 = numOnsetNacta2017Male[:10]
    numOnsetNacta2017Male1 = numOnsetNacta2017Male[10:20]
    numOnsetNacta2017Male2 = numOnsetNacta2017Male[20:30]
    numOnsetNacta2017Male3 = numOnsetNacta2017Male[30:]

    numOnsetNacta2017Fem0 = numOnsetNacta2017Fem[:10]
    numOnsetNacta2017Fem1 = numOnsetNacta2017Fem[10:]

    numOnsetNactaMale0 = numOnsetNactaMale[:10]
    numOnsetNactaMale1 = numOnsetNactaMale[10:]

    numOnsetNactaFem0 = numOnsetNactaFem[:10]
    numOnsetNactaFem1 = numOnsetNactaFem[10:]

    # obtain the indices of test recordings
    print('test 0 nacta 2017 male number', testRecordings(numOnsetNacta2017Male0, 0.2)) # (1,7)
    print('test 1 nacta 2017 male number', testRecordings(numOnsetNacta2017Male1, 0.2)) # (11,16)
    print('test 2 nacta 2017 male number', testRecordings(numOnsetNacta2017Male2, 0.2)) # (23,28)
    print('test 3 nacta 2017 male number', testRecordings(numOnsetNacta2017Male3, 0.2)) # (31,34,35,39)
    print(sum(itemgetter(*[1,7,11,16,23,28,31,34,35,39])(numOnsetNacta2017Male))/float(sum(numOnsetNacta2017Male)))

    print('test 0 nacta 2017 female number', testRecordings(numOnsetNacta2017Fem0, 0.2)) # (0,1,2,7)
    print('test 1 nacta 2017 female number', testRecordings(numOnsetNacta2017Fem1, 0.2)) # (13,14)
    print(sum(itemgetter(*[0,1,2,7,13,14])(numOnsetNacta2017Fem))/float(sum(numOnsetNacta2017Fem)))

    print('test 0 nacta male number', testRecordings(numOnsetNactaMale0, 0.2)) # (0,8)
    print('test 1 nacta male number', testRecordings(numOnsetNactaMale1, 0.2)) # (16)
    print(sum(itemgetter(*[0,8,16])(numOnsetNactaMale))/float(sum(numOnsetNactaMale)))

    print('test 0 nacta female number', testRecordings(numOnsetNactaFem0, 0.2)) # (1,9)
    print('test 1 nacta female number', testRecordings(numOnsetNactaFem1, 0.2)) # (10)
    print(sum(itemgetter(*[1,9,10])(numOnsetNactaFem))/float(sum(numOnsetNactaFem)))

    recordingsTestNacta2017Male = [[listOnsetNacta2017Male[ii][0], listOnsetNacta2017Male[ii][1]] for ii in (1,7,11,16,23,28,31,34,35,39)]
    recordingsTrainNacta2017Male = [[listOnsetNacta2017Male[ii][0], listOnsetNacta2017Male[ii][1]] for ii in range(len(listOnsetNacta2017Male)) if ii not in (1,7,11,16,23,28,31,34,35,39)]
    # numTestQmMale = [lineOnsetQmMale[ii][1] for ii in (0, 1, 7, 8)]

    recordingsTestNacta2017Fem = [[listOnsetNacta2017Fem[ii][0], listOnsetNacta2017Fem[ii][1]] for ii in
                                   (0,1,2,7,13,14)]
    recordingsTrainNacta2017Fem = [[listOnsetNacta2017Fem[ii][0], listOnsetNacta2017Fem[ii][1]] for ii in
                                    range(len(listOnsetNacta2017Fem)) if ii not in (0,1,2,7,13,14)]
    # numTestQmFem = [lineOnsetQmFem[ii][1] for ii in (4, 8)]

    recordingsTestNactaMale = [[listOnsetNactaMale[ii][0], listOnsetNactaMale[ii][1]] for ii in
                                  (0,8,16)]
    recordingsTrainNactaMale = [[listOnsetNactaMale[ii][0], listOnsetNactaMale[ii][1]] for ii in
                                    range(len(listOnsetNactaMale)) if ii not in (0,8,16)]
    # numTestLon = [lineOnsetLon[ii][1] for ii in (2,)]

    recordingsTestNactaFem = [[listOnsetNactaFem[ii][0], listOnsetNactaFem[ii][1]] for ii in
                               (1,9,10)]
    recordingsTrainNactaFem = [[listOnsetNactaFem[ii][0], listOnsetNactaFem[ii][1]] for ii in
                                range(len(listOnsetNactaFem)) if ii not in (1,9,10)]
    # numTestBcn = [lineOnsetBcn[ii][1] for ii in (4, 5)]

    return recordingsTestNacta2017Male+recordingsTestNacta2017Fem, recordingsTestNactaMale+recordingsTestNactaFem, \
           recordingsTrainNacta2017Male+recordingsTrainNacta2017Fem, recordingsTrainNactaMale+recordingsTrainNactaFem

def getTestTrainRecordingsNacta2017Artist():
    """
    partition train test for each artist
    :return:
    """

    """
    list_onset_nacta2017 = getBoundaryNumber(textgrid_path=nacta2017_textgrid_path, score_path=nacta2017_score_path)
    artists = list(set([list_recording[0] for list_recording in list_onset_nacta2017]))

    recordingsTestNacta = []
    recordingsNumTestNacta = 0
    recordingsTrainNacta = []
    recordingsNumTrainNacta = 0
    for ar in artists:
        print('partitioning', ar)

        list_artist = []
        for list_recording in list_onset_nacta2017:
            if list_recording[0] == ar:
                list_artist.append(list_recording)

        list_onset = [list_ar[2] for list_ar in list_artist]
        if len(list_onset) > 10:
            idx_test_set = []
            for ii_slo, sub_list_onset in enumerate(list(chunks(list_onset, 10))):
                sub_idx_test_set = testRecordings(sub_list_onset, 0.2)
                sub_idx_test_set = [its+ii_slo*10 for its in sub_idx_test_set]
                idx_test_set += sub_idx_test_set
        else:
            idx_test_set = testRecordings(list_onset, 0.2)

        recordingsTestNacta += [[list_artist[ii][0], list_artist[ii][1]] for ii in idx_test_set]
        recordingsTrainNacta += [[list_artist[ii][0], list_artist[ii][1]] for ii in range(len(list_artist)) if ii not in idx_test_set]

        recordingsNumTestNacta += sum([list_artist[ii][2] for ii in idx_test_set])
        recordingsNumTrainNacta += sum([list_artist[ii][2] for ii in range(len(list_artist)) if ii not in idx_test_set])
    """

    testNacta2017 = [['20170424SunYuZhu', 'daeh-Yi_sha_shi-Suo_lin_nang-nacta'],
     ['20170408SongRuoXuan', 'daxp-Su_san_li-Su_san_qi-nacta'],
     ['20170327LiaoJiaNi', 'lsxp-Jiang_shen_er-San_jia_dian-nacta'],
     ['20170425SunYuZhu', 'daeh-Wei_kai_yan-Dou_e_yuan-nacta'],
     ['20170519LongTianMing', 'lseh-Wei_guo_jia-Hong_yang_dong-ustb'],
     ['20170519LongTianMing', 'lseh-Zi_na_ri-Hong_yang_dong-ustb'],
     ['20170519XuJingWei', 'lseh-Wei_kai_yan-Rang_xu_zhou-renmin'],
     ['20170519XuJingWei', 'lsxp-Huai_nan_wang-Huai_he_ying-renmin'],
     ['20170418TianHao', 'lseh-Lao_zhang_bu-Wu_pen_ji01-nacta'],
     ['20170418TianHao', 'lseh-Niang_zi_bu-Sou_gu_jiu02-nacta'],
     ['20170418TianHao', 'lseh-Niang_zi_bu-Sou_gu_jiu03-nacta'],
     ['20170418TianHao', 'lseh-Xin_zhong_you-Wen_zhao_guan01-nacta'],
     ['20170418TianHao', 'lseh-Xin_zhong_you-Wen_zhao_guan03-nacta'],
     ['20170418TianHao', 'lseh-Yi_lun_ming-Wen_zhao_guan01-nacta'],
     ['20170418TianHao', 'lsxp-Liang_guo_jiao-Shi_jie_ting02-nacta'],
     ['20170418TianHao', 'lsxp-Wo_ben_shi-Kong_cheng_ji-nacta'],
     ['20170506LiuHaiLin', 'daeh-Wang_chun_e-San_niang_jiao-ustb'],
     ['20170506LiuHaiLin', 'daxp-Qiao_lou_shang-Huang_shan_lei-ustb']]
    trainNacta2017 = [['20170424SunYuZhu', 'daxp-Dang_ri_li-Suo_lin_nang-nacta'],
      ['20170424SunYuZhu', 'daxp-Er_ting_de-Suo_lin_nang_take1-nacta'],
      ['20170424SunYuZhu', 'daxp-Er_ting_de-Suo_lin_nang_take2-nacta'],
      ['20170424SunYuZhu', 'daxp-Er_ting_de-Suo_lin_nang_take3-nacta'],
      ['20170424SunYuZhu', 'daxp-Zhe_cai_shi-Suo_lin_nang-nacta'],
      ['20170424SunYuZhu', 'daxp-Zhe_cai_shi-Suo_lin_nang_first_half-nacta'],
      ['20170408SongRuoXuan', 'daeh-Yang_yu_huan-Tai_zhen_wai-nacta'],
      ['20170408SongRuoXuan', 'danbz-Kan_dai_wang-Ba_wang_bie-nacta'],
      ['20170408SongRuoXuan', 'daspd-Hai_dao_bing-Gui_fei_zui-nacta'],
      ['20170408SongRuoXuan', 'daxp-Lao_die_die-Yu_zhou_feng-nacta'],
      ['20170327LiaoJiaNi', 'lseh-Niang_zi_bu-Sou_gu_jiu-nacta'],
      ['20170327LiaoJiaNi', 'lseh-Yi_lun_ming-Wen_zhao_guan-nacta'],
      ['20170327LiaoJiaNi', 'lsxp-Xi_ri_li-Zhu_lian_zhai-nacta'],
      ['20170327LiaoJiaNi', 'lsxp-Yi_ma_li-Wu_jia_po-nacta'],
      ['20170425SunYuZhu', 'daxp-Chui_qiu_ting-Suo_lin_nang-nacta'],
      ['20170425SunYuZhu', 'daxp-Chui_qiu_ting-Suo_lin_nang_first_line-nacta'],
      ['20170519LongTianMing', 'lseh-Tan_yang_jia-Hong_yang_dong-ustb'],
      ['20170519LongTianMing', 'lseh-Yi_lun_ming-Zhuo_fang_cao-ustb'],
      ['20170519LongTianMing', 'lsxp-Ting_ta_yan-Zhuo_fang_cao-ustb'],
      ['20170519LongTianMing', 'lsxp-Wo_ben_shi-Kong_cheng_ji-ustb'],
      ['20170519LongTianMing', 'lsxp-Wo_zheng_zai-Kong_cheng_ji-ustb'],
      ['20170519XuJingWei', 'lseh-Jin_zhong_xiang-Shang_tian_tai-renmin'],
      ['20170519XuJingWei', 'lseh-Wei_guo_jia-Hong_yang_dong-renmin'],
      ['20170519XuJingWei', 'lsxp-Jiang_shen_er-San_jia_dian-renmin'],
      ['20170519XuJingWei', 'lsxp-Wo_ben_shi-Kong_cheng_ji-renmin'],
      ['20170519XuJingWei', 'lsxp-Wo_zheng_zai-Kong_cheng_ji-renmin'],
      ['20170519XuJingWei', 'lsxp-Yi_ma_li-Wu_jia_po-renmin'],
      ['20170418TianHao', 'lseh-Jin_zhong_xiang-Shang_tian_tai01-nacta'],
      ['20170418TianHao', 'lseh-Jin_zhong_xiang-Shang_tian_tai02-nacta'],
      ['20170418TianHao', 'lseh-Lao_zhang_bu-Wu_pen_ji02-nacta'],
      ['20170418TianHao', 'lseh-Niang_zi_bu-Sou_gu_jiu01-nacta'],
      ['20170418TianHao', 'lseh-Tan_yang_jia-Hong_yang_dong-nacta'],
      ['20170418TianHao', 'lseh-Wei_guo_jia-Hong_yang_dong01-nacta'],
      ['20170418TianHao', 'lseh-Wei_guo_jia-Hong_yang_dong02-nacta'],
      ['20170418TianHao', 'lseh-Xin_zhong_you-Wen_zhao_guan02-nacta'],
      ['20170418TianHao', 'lseh-Yi_lun_ming-Wen_zhao_guan02-nacta'],
      ['20170418TianHao', 'lsxp-Jiang_shen_er-San_jia_dian-nacta'],
      ['20170418TianHao', 'lsxp-Liang_guo_jiao-Shi_jie_ting01-nacta'],
      ['20170418TianHao', 'lsxp-Ting_ta_yan-Zhuo_fang_cao01-nacta'],
      ['20170418TianHao', 'lsxp-Ting_ta_yan-Zhuo_fang_cao02-nacta'],
      ['20170418TianHao', 'lsxp-Ting_ta_yan-Zhuo_fang_cao03-nacta'],
      ['20170418TianHao', 'lsxp-Wo_zheng_zai-Kong_cheng_ji01-nacta'],
      ['20170418TianHao', 'lsxp-Wo_zheng_zai-Kong_cheng_ji02-nacta'],
      ['20170418TianHao', 'lsxp-Xi_ri_li-Zhu_lian_zhai-nacta'],
      ['20170506LiuHaiLin', 'daeh-Wei_kai_yan-Dou_e_yuan-ustb'],
      ['20170506LiuHaiLin', 'daxp-Chun_qiu_ting-Suo_lin_nang-ustb'],
      ['20170506LiuHaiLin', 'daxp-Dang_ri_li-Suo_lin_nang-ustb'],
      ['20170506LiuHaiLin', 'daxp-Yi_sha_shi-Suo_lin_nang-ustb']]
    # print(recordingsTestNacta, recordingsNumTestNacta)
    # print(recordingsTrainNacta, recordingsNumTrainNacta)

    return testNacta2017, trainNacta2017

def getTestTrainRecordingsArtistAlbumFilter():
    """

    list_onset_nacta2017 = getBoundaryNumber(textgrid_path=nacta2017_textgrid_path, score_path=nacta2017_score_path)
    list_onset_nacta = getBoundaryNumber(textgrid_path=nacta_textgrid_path, score_path=nacta_score_path)

    onset_number_nacta2017_train = 0
    onset_number_nacta2017_test = 0
    onset_number_nacta_train = 0
    onset_number_nacta_test = 0
    testNacta = []
    trainNacta = []
    testNacta2017 = []
    trainNacta2017 = []

    for lon in list_onset_nacta2017:
        if lon[0] == '20170327LiaoJiaNi':
            onset_number_nacta2017_test += lon[2]
            testNacta2017.append([lon[0], lon[1]])
        else:
            onset_number_nacta2017_train += lon[2]
            trainNacta2017.append([lon[0], lon[1]])

    for lon in list_onset_nacta:
        if '-lon' in lon[1] or '-upf' in lon[1]:
            onset_number_nacta_test += lon[2]
            testNacta.append([lon[0], lon[1]])
        else:
            onset_number_nacta_train += lon[2]
            trainNacta.append([lon[0], lon[1]])

    # print(onset_number_nacta2017_train, onset_number_nacta2017_test)
    # print(onset_number_nacta_train, onset_number_nacta_test)
    #
    # print(float(onset_number_nacta2017_test+onset_number_nacta_test)/
    #       ((onset_number_nacta2017_train+onset_number_nacta_train)+(onset_number_nacta2017_test+onset_number_nacta_test)))
    """
    # yes or no means that if there is the score
    testNacta = [['danAll', 'daeh-Yang_Yu_huan-Tai_zhen_wai_zhuan-lon'], # yes
                 ['danAll', 'dagbz-Feng_xiao_xiao-Yang_men_nv_jiang-lon'], # no
                ['danAll', 'daspd-Du_shou_kong-Wang_jiang_ting-upf'], # yes
                 ['danAll', 'daspd-Hai_dao_bing-Gui_fei_zui_jiu01-lon'], # yes
                ['danAll', 'daxp-Guan_Shi_yin-Tian_nv_san_hua-lon'], # yes
                 ['danAll', 'daxp-Meng_ting_de-Mu_Gui_ying_gua_shuai01-upf'], # yes
                ['laosheng', 'lseh-Wei_guo_jia-Hong_yang_dong01-lon'], # yes
                 ['laosheng', 'lsxp-Guo_liao_yi-Wen_zhao_guan01-upf'], # no
                ['laosheng', 'lsxp-Huai_nan_wang-Huai_he_ying01-lon'], # yes
                 ['laosheng', 'lsxp-Jiang_shen_er-San_jia_dian01-1-upf'], # yes
                ['laosheng', 'lsxp-Jiang_shen_er-San_jia_dian01-2-upf'], # yes
                 ['laosheng', 'lsxp-Wo_zheng_zai-Kong_cheng_ji01-upf']] # yes

    testNacta2017 = [['20170327LiaoJiaNi', 'lseh-Niang_zi_bu-Sou_gu_jiu-nacta'], # yes
                     ['20170327LiaoJiaNi', 'lseh-Yi_lun_ming-Wen_zhao_guan-nacta'], # yes
                     ['20170327LiaoJiaNi', 'lsxp-Jiang_shen_er-San_jia_dian-nacta'], # yes
                     ['20170327LiaoJiaNi', 'lsxp-Xi_ri_li-Zhu_lian_zhai-nacta'], # yes
                     ['20170327LiaoJiaNi', 'lsxp-Yi_ma_li-Wu_jia_po-nacta']] # yes

    trainNacta = [['danAll', 'dafeh-Bi_yun_tian-Xi_xiang_ji01-qm'],
                  ['danAll', 'danbz-Bei_jiu_chan-Chun_gui_men01-qm'],
                ['danAll', 'danbz-Kan_dai_wang-Ba_wang_bie_ji01-qm'],
                  ['danAll', 'daspd-Hai_dao_bing-Gui_fei_zui_jiu02-qm'],
                    ['danAll', 'daxp-Chun_qiu_ting-Suo_lin_nang01-qm'],
                  ['danAll', 'daxp-Jiao_Zhang_sheng-Hong_niang01-qm'],
                    ['danAll', 'daxp-Jiao_Zhang_sheng-Hong_niang04-qm'],
                  ['danAll', 'daxp-Meng_ting_de-Mu_Gui_ying_gua_shuai02-qm'],
                ['danAll', 'daxp-Meng_ting_de-Mu_Gui_ying_gua_shuai04-qm'],
                  ['danAll', 'daxp-Zhe_cai_shi-Suo_lin_nang01-qm'],
                    ['laosheng', 'lseh-Tan_Yang_jia-Hong_yang_dong-qm'],
                  ['laosheng', 'lseh-Wei_guo_jia-Hong_yang_dong02-qm'],
                    ['laosheng', 'lseh-Wo_ben_shi-Qiong_lin_yan-qm'],
                  ['laosheng', 'lseh-Yi_lun_ming-Wen_zhao_guan-qm'],
                    ['laosheng', 'lseh-Zi_na_ri-Hong_yang_dong-qm'],
                  ['laosheng', 'lsxp-Guo_liao_yi-Wen_zhao_guan02-qm'],
                    ['laosheng', 'lsxp-Huai_nan_wang-Huai_he_ying02-qm'],
                  ['laosheng', 'lsxp-Jiang_shen_er-San_jia_dian02-qm'],
                    ['laosheng', 'lsxp-Qian_bai_wan-Si_lang_tang_mu01-qm'],
                  ['laosheng', 'lsxp-Quan_qian_sui-Gan_lu_si-qm'],
                    ['laosheng', 'lsxp-Shi_ye_shuo-Ding_jun_shan-qm'],
                  ['laosheng', 'lsxp-Wo_ben_shi-Kong_cheng_ji-qm'],
                    ['laosheng', 'lsxp-Wo_zheng_zai-Kong_cheng_ji04-qm'],
                  ['laosheng', 'lsxp-Xi_ri_you-Zhu_lian_zhai-qm']]

    trainNacta2017 = [['20170408SongRuoXuan', 'daeh-Yang_yu_huan-Tai_zhen_wai-nacta'],
     ['20170408SongRuoXuan', 'danbz-Kan_dai_wang-Ba_wang_bie-nacta'],
     ['20170408SongRuoXuan', 'daspd-Hai_dao_bing-Gui_fei_zui-nacta'],
     ['20170408SongRuoXuan', 'daxp-Lao_die_die-Yu_zhou_feng-nacta'],
     ['20170408SongRuoXuan', 'daxp-Su_san_li-Su_san_qi-nacta'],
     ['20170418TianHao', 'lseh-Jin_zhong_xiang-Shang_tian_tai01-nacta'],
     ['20170418TianHao', 'lseh-Jin_zhong_xiang-Shang_tian_tai02-nacta'],
     ['20170418TianHao', 'lseh-Lao_zhang_bu-Wu_pen_ji01-nacta'],
     ['20170418TianHao', 'lseh-Lao_zhang_bu-Wu_pen_ji02-nacta'],
     ['20170418TianHao', 'lseh-Niang_zi_bu-Sou_gu_jiu01-nacta'],
     ['20170418TianHao', 'lseh-Niang_zi_bu-Sou_gu_jiu02-nacta'],
     ['20170418TianHao', 'lseh-Niang_zi_bu-Sou_gu_jiu03-nacta'],
     ['20170418TianHao', 'lseh-Tan_yang_jia-Hong_yang_dong-nacta'],
     ['20170418TianHao', 'lseh-Wei_guo_jia-Hong_yang_dong01-nacta'],
     ['20170418TianHao', 'lseh-Wei_guo_jia-Hong_yang_dong02-nacta'],
     ['20170418TianHao', 'lseh-Xin_zhong_you-Wen_zhao_guan01-nacta'],
     ['20170418TianHao', 'lseh-Xin_zhong_you-Wen_zhao_guan02-nacta'],
     ['20170418TianHao', 'lseh-Xin_zhong_you-Wen_zhao_guan03-nacta'],
     ['20170418TianHao', 'lseh-Yi_lun_ming-Wen_zhao_guan01-nacta'],
     ['20170418TianHao', 'lseh-Yi_lun_ming-Wen_zhao_guan02-nacta'],
     ['20170418TianHao', 'lsxp-Jiang_shen_er-San_jia_dian-nacta'],
     ['20170418TianHao', 'lsxp-Liang_guo_jiao-Shi_jie_ting01-nacta'],
     ['20170418TianHao', 'lsxp-Liang_guo_jiao-Shi_jie_ting02-nacta'],
     ['20170418TianHao', 'lsxp-Ting_ta_yan-Zhuo_fang_cao01-nacta'],
     ['20170418TianHao', 'lsxp-Ting_ta_yan-Zhuo_fang_cao02-nacta'],
     ['20170418TianHao', 'lsxp-Ting_ta_yan-Zhuo_fang_cao03-nacta'],
     ['20170418TianHao', 'lsxp-Wo_ben_shi-Kong_cheng_ji-nacta'],
     ['20170418TianHao', 'lsxp-Wo_zheng_zai-Kong_cheng_ji01-nacta'],
     ['20170418TianHao', 'lsxp-Wo_zheng_zai-Kong_cheng_ji02-nacta'],
     ['20170418TianHao', 'lsxp-Xi_ri_li-Zhu_lian_zhai-nacta'],
     ['20170424SunYuZhu', 'daeh-Yi_sha_shi-Suo_lin_nang-nacta'],
     ['20170424SunYuZhu', 'daxp-Dang_ri_li-Suo_lin_nang-nacta'],
     ['20170424SunYuZhu', 'daxp-Er_ting_de-Suo_lin_nang_take1-nacta'],
     ['20170424SunYuZhu', 'daxp-Er_ting_de-Suo_lin_nang_take2-nacta'],
     ['20170424SunYuZhu', 'daxp-Er_ting_de-Suo_lin_nang_take3-nacta'],
     ['20170424SunYuZhu', 'daxp-Zhe_cai_shi-Suo_lin_nang-nacta'],
     ['20170424SunYuZhu', 'daxp-Zhe_cai_shi-Suo_lin_nang_first_half-nacta'],
     ['20170425SunYuZhu', 'daeh-Wei_kai_yan-Dou_e_yuan-nacta'],
     ['20170425SunYuZhu', 'daxp-Chui_qiu_ting-Suo_lin_nang-nacta'],
     ['20170425SunYuZhu', 'daxp-Chui_qiu_ting-Suo_lin_nang_first_line-nacta'],
     ['20170506LiuHaiLin', 'daeh-Wang_chun_e-San_niang_jiao-ustb'],
     ['20170506LiuHaiLin', 'daeh-Wei_kai_yan-Dou_e_yuan-ustb'],
     ['20170506LiuHaiLin', 'daxp-Chun_qiu_ting-Suo_lin_nang-ustb'],
     ['20170506LiuHaiLin', 'daxp-Dang_ri_li-Suo_lin_nang-ustb'],
     ['20170506LiuHaiLin', 'daxp-Qiao_lou_shang-Huang_shan_lei-ustb'],
     ['20170506LiuHaiLin', 'daxp-Yi_sha_shi-Suo_lin_nang-ustb'],
     ['20170519LongTianMing', 'lseh-Tan_yang_jia-Hong_yang_dong-ustb'],
     ['20170519LongTianMing', 'lseh-Wei_guo_jia-Hong_yang_dong-ustb'],
     ['20170519LongTianMing', 'lseh-Yi_lun_ming-Zhuo_fang_cao-ustb'],
     ['20170519LongTianMing', 'lseh-Zi_na_ri-Hong_yang_dong-ustb'],
     ['20170519LongTianMing', 'lsxp-Ting_ta_yan-Zhuo_fang_cao-ustb'],
     ['20170519LongTianMing', 'lsxp-Wo_ben_shi-Kong_cheng_ji-ustb'],
     ['20170519LongTianMing', 'lsxp-Wo_zheng_zai-Kong_cheng_ji-ustb'],
     ['20170519XuJingWei', 'lseh-Jin_zhong_xiang-Shang_tian_tai-renmin'],
     ['20170519XuJingWei', 'lseh-Wei_guo_jia-Hong_yang_dong-renmin'],
     ['20170519XuJingWei', 'lseh-Wei_kai_yan-Rang_xu_zhou-renmin'],
     ['20170519XuJingWei', 'lsxp-Huai_nan_wang-Huai_he_ying-renmin'],
     ['20170519XuJingWei', 'lsxp-Jiang_shen_er-San_jia_dian-renmin'],
     ['20170519XuJingWei', 'lsxp-Wo_ben_shi-Kong_cheng_ji-renmin'],
     ['20170519XuJingWei', 'lsxp-Wo_zheng_zai-Kong_cheng_ji-renmin'],
     ['20170519XuJingWei', 'lsxp-Yi_ma_li-Wu_jia_po-renmin']]

    return testNacta2017, testNacta, trainNacta2017, trainNacta


def getTestTrainRecordingsJoint():
    # trainNacta2017, trainNacta for syllable and phoneme joint estimation and HSMM alignment acoustic model
    trainNacta2017 = [['20170418TianHao', 'lseh-Tan_yang_jia-Hong_yang_dong-nacta']]  # yes # pro

    trainNacta = [['danAll', 'dafeh-Bi_yun_tian-Xi_xiang_ji01-qm'],
                  ['danAll', 'danbz-Bei_jiu_chan-Chun_gui_men01-qm'],
                  ['danAll', 'danbz-Kan_dai_wang-Ba_wang_bie_ji01-qm'],
                  ['danAll', 'daspd-Hai_dao_bing-Gui_fei_zui_jiu02-qm'],
                  ['danAll', 'daxp-Chun_qiu_ting-Suo_lin_nang01-qm'],
                  ['danAll', 'daxp-Jiao_Zhang_sheng-Hong_niang01-qm'],
                  ['danAll', 'daxp-Meng_ting_de-Mu_Gui_ying_gua_shuai04-qm'],
                  ['danAll', 'daxp-Zhe_cai_shi-Suo_lin_nang01-qm'],
                  ['laosheng', 'lseh-Tan_Yang_jia-Hong_yang_dong-qm'], # amateur
                  ['laosheng', 'lseh-Wo_ben_shi-Qiong_lin_yan-qm'], # amateur
                  ['laosheng', 'lseh-Yi_lun_ming-Wen_zhao_guan-qm'], # amateur
                  ['laosheng', 'lseh-Zi_na_ri-Hong_yang_dong-qm'], # amateur
                  ['laosheng', 'lsxp-Guo_liao_yi-Wen_zhao_guan02-qm'],
                  ['laosheng', 'lsxp-Huai_nan_wang-Huai_he_ying02-qm'],
                  ['laosheng', 'lsxp-Jiang_shen_er-San_jia_dian02-qm'],
                  ['laosheng', 'lsxp-Qian_bai_wan-Si_lang_tang_mu01-qm'], # amateur
                  ['laosheng', 'lsxp-Quan_qian_sui-Gan_lu_si-qm'],  # amateur
                  ['laosheng', 'lsxp-Shi_ye_shuo-Ding_jun_shan-qm'], # amateur
                  ['laosheng', 'lsxp-Wo_ben_shi-Kong_cheng_ji-qm'], # amateur
                  ['laosheng', 'lsxp-Xi_ri_you-Zhu_lian_zhai-qm'], # amateur
                    ]

    return trainNacta2017, trainNacta


def getTestRecordingsScoreDurCorrectionArtistAlbumFilter():
    """
    For the experiment of score duration correction by audio to score alignment
    These audios all have xml scores in Scores/artistAlbumFilterAudioScoreAlignment
    :return: testNacta2017, testNacta
    """
    testNacta = [['danAll', 'daeh-Yang_Yu_huan-Tai_zhen_wai_zhuan-lon'],  # yes
                 ['danAll', 'daspd-Du_shou_kong-Wang_jiang_ting-upf'],  # yes
                 ['danAll', 'daspd-Hai_dao_bing-Gui_fei_zui_jiu01-lon'],  # yes
                 ['danAll', 'daxp-Guan_Shi_yin-Tian_nv_san_hua-lon'],  # yes
                 ['danAll', 'daxp-Meng_ting_de-Mu_Gui_ying_gua_shuai01-upf'],  # yes
                 ['laosheng', 'lseh-Wei_guo_jia-Hong_yang_dong01-lon'],  # yes
                 ['laosheng', 'lsxp-Huai_nan_wang-Huai_he_ying01-lon'],  # yes
                 ['laosheng', 'lsxp-Jiang_shen_er-San_jia_dian01-1-upf'],  # yes
                 ['laosheng', 'lsxp-Jiang_shen_er-San_jia_dian01-2-upf'],  # yes
                 ['laosheng', 'lsxp-Wo_zheng_zai-Kong_cheng_ji01-upf']]  # yes

    testNacta2017 = [['20170327LiaoJiaNi', 'lseh-Niang_zi_bu-Sou_gu_jiu-nacta'],  # yes
                     ['20170327LiaoJiaNi', 'lseh-Yi_lun_ming-Wen_zhao_guan-nacta'],  # yes
                     ['20170327LiaoJiaNi', 'lsxp-Jiang_shen_er-San_jia_dian-nacta'],  # yes
                     ['20170327LiaoJiaNi', 'lsxp-Xi_ri_li-Zhu_lian_zhai-nacta'],  # yes
                     ['20170327LiaoJiaNi', 'lsxp-Yi_ma_li-Wu_jia_po-nacta']]  # yes
    return testNacta2017, testNacta

def getBoundaryNumberTestTrainHelper(records,list_onset_all):
    num = 0
    for tr in records:
        print(tr)
        for loa in list_onset_all:
            if tr == loa[:2]:
                print('find syllable number', tr)
                num += loa[2]
    return num

def getBoundaryNumberTestTrain(testNacta2017, testNacta, trainNacta2017, trainNacta):
    # check test train syllable numbers
    list_onset_nacta2017 = getBoundaryNumber(textgrid_path=nacta2017_textgrid_path, score_path=nacta2017_score_path)
    list_onset_nacta = getBoundaryNumber(textgrid_path=nacta_textgrid_path, score_path=nacta_score_path)

    list_onset_all = list_onset_nacta2017+list_onset_nacta

    testRecords = testNacta+testNacta2017
    trainRecords = trainNacta+trainNacta2017

    numTest = getBoundaryNumberTestTrainHelper(testRecords, list_onset_all)
    numTrain = getBoundaryNumberTestTrainHelper(trainRecords, list_onset_all)

    print(numTest, numTrain)


def getTestTrainRecordingsArtist():
    """
    nacta 2017 artist partition + old dataset artist partition
    :return:
    """
    testNacta2017, trainNacta2017 = getTestTrainRecordingsNacta2017Artist()

    # calculated from the master branch trainTestSeparation.py
    testNacta = [['danAll', 'daxp-Zhe_cai_shi-Suo_lin_nang01-qm'], ['danAll', 'dafeh-Bi_yun_tian-Xi_xiang_ji01-qm'],
                     ['danAll', 'daspd-Hai_dao_bing-Gui_fei_zui_jiu02-qm'],
                     ['laosheng', 'lseh-Wei_guo_jia-Hong_yang_dong02-qm'],
                     ['laosheng', 'lseh-Wo_ben_shi-Qiong_lin_yan-qm'],
                     ['laosheng', 'lseh-Yi_lun_ming-Wen_zhao_guan-qm'],
                     ['laosheng', 'lsxp-Huai_nan_wang-Huai_he_ying02-qm'],
                     ['danAll', 'dagbz-Feng_xiao_xiao-Yang_men_nv_jiang-lon'],
                     ['laosheng', 'lsxp-Huai_nan_wang-Huai_he_ying01-lon'],
                     ['danAll', 'daspd-Du_shou_kong-Wang_jiang_ting-upf'],
                     ['laosheng', 'lsxp-Wo_zheng_zai-Kong_cheng_ji01-upf'],
                     ['laosheng', 'lsxp-Guo_liao_yi-Wen_zhao_guan02-qm'],
                     ['danAll', 'daxp-Jiao_Zhang_sheng-Hong_niang01-qm']]
    trainNacta = [['danAll', 'danbz-Kan_dai_wang-Ba_wang_bie_ji01-qm'],
                      ['danAll', 'daxp-Meng_ting_de-Mu_Gui_ying_gua_shuai04-qm'],
                      ['danAll', 'daxp-Jiao_Zhang_sheng-Hong_niang04-qm'],
                      ['danAll', 'daxp-Chun_qiu_ting-Suo_lin_nang01-qm'],
                      ['danAll', 'danbz-Bei_jiu_chan-Chun_gui_men01-qm'],
                      ['danAll', 'daxp-Meng_ting_de-Mu_Gui_ying_gua_shuai02-qm'],
                      ['laosheng', 'lseh-Tan_Yang_jia-Hong_yang_dong-qm'],
                      ['laosheng', 'lseh-Zi_na_ri-Hong_yang_dong-qm'],
                      ['laosheng', 'lsxp-Xi_ri_you-Zhu_lian_zhai-qm'], ['laosheng', 'lsxp-Quan_qian_sui-Gan_lu_si-qm'],
                      ['laosheng', 'lsxp-Shi_ye_shuo-Ding_jun_shan-qm'],
                      ['laosheng', 'lsxp-Wo_ben_shi-Kong_cheng_ji-qm'],
                      ['laosheng', 'lsxp-Wo_zheng_zai-Kong_cheng_ji04-qm'],
                      ['laosheng', 'lsxp-Qian_bai_wan-Si_lang_tang_mu01-qm'],
                      ['danAll', 'daxp-Guan_Shi_yin-Tian_nv_san_hua-lon'],
                      ['danAll', 'daeh-Yang_Yu_huan-Tai_zhen_wai_zhuan-lon'],
                      ['danAll', 'daspd-Hai_dao_bing-Gui_fei_zui_jiu01-lon'],
                      ['laosheng', 'lseh-Wei_guo_jia-Hong_yang_dong01-lon'],
                      ['danAll', 'daxp-Meng_ting_de-Mu_Gui_ying_gua_shuai01-upf'],
                      ['laosheng', 'lsxp-Jiang_shen_er-San_jia_dian01-1-upf'],
                      ['laosheng', 'lsxp-Jiang_shen_er-San_jia_dian01-2-upf'],
                      ['laosheng', 'lsxp-Guo_liao_yi-Wen_zhao_guan01-upf'],
                      ['laosheng', 'lsxp-Jiang_shen_er-San_jia_dian02-qm']]


    return testNacta2017, testNacta, trainNacta2017, trainNacta

def getTestTrainRecordingsNactaISMIR():
    """
    coherent to ismir 2017 paper datasplit
    :return:
    """
    # trainNacta2017 = [['20170327LiaoJiaNi', 'lseh-Niang_zi_bu-Sou_gu_jiu-nacta'],
    #                   ['20170327LiaoJiaNi', 'lsxp-Jiang_shen_er-San_jia_dian-nacta'],
    #                   ['20170327LiaoJiaNi', 'lsxp-Xi_ri_li-Zhu_lian_zhai-nacta'],
    #                   ['20170327LiaoJiaNi', 'lsxp-Yi_ma_li-Wu_jia_po-nacta'],
    #                   ['20170418TianHao', 'lseh-Jin_zhong_xiang-Shang_tian_tai01-nacta'],
    #                   ['20170418TianHao', 'lseh-Jin_zhong_xiang-Shang_tian_tai02-nacta'],
    #                   ['20170418TianHao', 'lseh-Lao_zhang_bu-Wu_pen_ji02-nacta'],
    #                   ['20170418TianHao', 'lseh-Niang_zi_bu-Sou_gu_jiu01-nacta'],
    #                   ['20170418TianHao', 'lseh-Niang_zi_bu-Sou_gu_jiu02-nacta'],
    #                   ['20170418TianHao', 'lseh-Tan_yang_jia-Hong_yang_dong-nacta'],
    #                   ['20170418TianHao', 'lseh-Wei_guo_jia-Hong_yang_dong01-nacta'],
    #                   ['20170418TianHao', 'lseh-Wei_guo_jia-Hong_yang_dong02-nacta'],
    #                   ['20170418TianHao', 'lseh-Xin_zhong_you-Wen_zhao_guan01-nacta'],
    #                   ['20170418TianHao', 'lseh-Xin_zhong_you-Wen_zhao_guan03-nacta'],
    #                   ['20170418TianHao', 'lseh-Yi_lun_ming-Wen_zhao_guan01-nacta'],
    #                   ['20170418TianHao', 'lseh-Yi_lun_ming-Wen_zhao_guan02-nacta'],
    #                   ['20170418TianHao', 'lsxp-Jiang_shen_er-San_jia_dian-nacta'],
    #                   ['20170418TianHao', 'lsxp-Liang_guo_jiao-Shi_jie_ting01-nacta'],
    #                   ['20170418TianHao', 'lsxp-Liang_guo_jiao-Shi_jie_ting02-nacta'],
    #                   ['20170418TianHao', 'lsxp-Ting_ta_yan-Zhuo_fang_cao02-nacta'],
    #                   ['20170418TianHao', 'lsxp-Ting_ta_yan-Zhuo_fang_cao03-nacta'],
    #                   ['20170418TianHao', 'lsxp-Wo_ben_shi-Kong_cheng_ji-nacta'],
    #                   ['20170418TianHao', 'lsxp-Wo_zheng_zai-Kong_cheng_ji01-nacta'],
    #                   ['20170418TianHao', 'lsxp-Xi_ri_li-Zhu_lian_zhai-nacta'],
    #                   ['20170519LongTianMing', 'lseh-Tan_yang_jia-Hong_yang_dong-ustb'],
    #                   ['20170519LongTianMing', 'lseh-Yi_lun_ming-Zhuo_fang_cao-ustb'],
    #                   ['20170519LongTianMing', 'lseh-Zi_na_ri-Hong_yang_dong-ustb'],
    #                   ['20170519LongTianMing', 'lsxp-Wo_zheng_zai-Kong_cheng_ji-ustb'],
    #                   ['20170519XuJingWei', 'lseh-Jin_zhong_xiang-Shang_tian_tai-renmin'],
    #                   ['20170519XuJingWei', 'lseh-Wei_guo_jia-Hong_yang_dong-renmin'],
    #                   ['20170519XuJingWei', 'lsxp-Huai_nan_wang-Huai_he_ying-renmin'],
    #                   ['20170519XuJingWei', 'lsxp-Jiang_shen_er-San_jia_dian-renmin'],
    #                   ['20170519XuJingWei', 'lsxp-Wo_ben_shi-Kong_cheng_ji-renmin'],
    #                   ['20170519XuJingWei', 'lsxp-Wo_zheng_zai-Kong_cheng_ji-renmin'],
    #                   ['20170519XuJingWei', 'lsxp-Yi_ma_li-Wu_jia_po-renmin'],
    #                   ['20170408SongRuoXuan', 'daxp-Lao_die_die-Yu_zhou_feng-nacta'],
    #                   ['20170408SongRuoXuan', 'daxp-Su_san_li-Su_san_qi-nacta'],
    #                   ['20170424SunYuZhu', 'daeh-Yi_sha_shi-Suo_lin_nang-nacta'],
    #                   ['20170424SunYuZhu', 'daxp-Dang_ri_li-Suo_lin_nang-nacta'],
    #                   ['20170424SunYuZhu', 'daxp-Er_ting_de-Suo_lin_nang_take2-nacta'],
    #                   ['20170424SunYuZhu', 'daxp-Er_ting_de-Suo_lin_nang_take3-nacta'],
    #                   ['20170424SunYuZhu', 'daxp-Zhe_cai_shi-Suo_lin_nang-nacta'],
    #                   ['20170424SunYuZhu', 'daxp-Zhe_cai_shi-Suo_lin_nang_first_half-nacta'],
    #                   ['20170425SunYuZhu', 'daeh-Wei_kai_yan-Dou_e_yuan-nacta'],
    #                   ['20170506LiuHaiLin', 'daeh-Wang_chun_e-San_niang_jiao-ustb'],
    #                   ['20170506LiuHaiLin', 'daeh-Wei_kai_yan-Dou_e_yuan-ustb'],
    #                   ['20170506LiuHaiLin', 'daxp-Chun_qiu_ting-Suo_lin_nang-ustb'],
    #                   ['20170506LiuHaiLin', 'daxp-Dang_ri_li-Suo_lin_nang-ustb'],
    #                   ['20170506LiuHaiLin', 'daxp-Qiao_lou_shang-Huang_shan_lei-ustb'],
    #                   ['20170506LiuHaiLin', 'daxp-Yi_sha_shi-Suo_lin_nang-ustb']]

    trainNacta2017 = []

    # [['20170327LiaoJiaNi', 'lseh-Niang_zi_bu-Sou_gu_jiu-nacta'],
    #  ['20170327LiaoJiaNi', 'lseh-Yi_lun_ming-Wen_zhao_guan-nacta'],
    #  ['20170327LiaoJiaNi', 'lsxp-Jiang_shen_er-San_jia_dian-nacta'],
    #  ['20170327LiaoJiaNi', 'lsxp-Xi_ri_li-Zhu_lian_zhai-nacta'], ['20170327LiaoJiaNi', 'lsxp-Yi_ma_li-Wu_jia_po-nacta'],
    #  ['20170408SongRuoXuan', 'daeh-Yang_yu_huan-Tai_zhen_wai-nacta'],
    #  ['20170408SongRuoXuan', 'danbz-Kan_dai_wang-Ba_wang_bie-nacta'],
    #  ['20170408SongRuoXuan', 'daspd-Hai_dao_bing-Gui_fei_zui-nacta'],
    #  ['20170408SongRuoXuan', 'daxp-Lao_die_die-Yu_zhou_feng-nacta'],
    #  ['20170408SongRuoXuan', 'daxp-Su_san_li-Su_san_qi-nacta'],
    #  ['20170418TianHao', 'lseh-Jin_zhong_xiang-Shang_tian_tai01-nacta'],
    #  ['20170418TianHao', 'lseh-Jin_zhong_xiang-Shang_tian_tai02-nacta'],
    #  ['20170418TianHao', 'lseh-Lao_zhang_bu-Wu_pen_ji01-nacta'],
    #  ['20170418TianHao', 'lseh-Lao_zhang_bu-Wu_pen_ji02-nacta'],
    #  ['20170418TianHao', 'lseh-Niang_zi_bu-Sou_gu_jiu01-nacta'],
    #  ['20170418TianHao', 'lseh-Niang_zi_bu-Sou_gu_jiu02-nacta'],
    #  ['20170418TianHao', 'lseh-Niang_zi_bu-Sou_gu_jiu03-nacta'],
    #  ['20170418TianHao', 'lseh-Tan_yang_jia-Hong_yang_dong-nacta'],
    #  ['20170418TianHao', 'lseh-Wei_guo_jia-Hong_yang_dong01-nacta'],
    #  ['20170418TianHao', 'lseh-Wei_guo_jia-Hong_yang_dong02-nacta'],
    #  ['20170418TianHao', 'lseh-Xin_zhong_you-Wen_zhao_guan01-nacta'],
    #  ['20170418TianHao', 'lseh-Xin_zhong_you-Wen_zhao_guan02-nacta'],
    #  ['20170418TianHao', 'lseh-Xin_zhong_you-Wen_zhao_guan03-nacta'],
    #  ['20170418TianHao', 'lseh-Yi_lun_ming-Wen_zhao_guan01-nacta'],
    #  ['20170418TianHao', 'lseh-Yi_lun_ming-Wen_zhao_guan02-nacta'],
    #  ['20170418TianHao', 'lsxp-Jiang_shen_er-San_jia_dian-nacta'],
    #  ['20170418TianHao', 'lsxp-Liang_guo_jiao-Shi_jie_ting01-nacta'],
    #  ['20170418TianHao', 'lsxp-Liang_guo_jiao-Shi_jie_ting02-nacta'],
    #  ['20170418TianHao', 'lsxp-Ting_ta_yan-Zhuo_fang_cao01-nacta'],
    #  ['20170418TianHao', 'lsxp-Ting_ta_yan-Zhuo_fang_cao02-nacta'],
    #  ['20170418TianHao', 'lsxp-Ting_ta_yan-Zhuo_fang_cao03-nacta'],
    #  ['20170418TianHao', 'lsxp-Wo_ben_shi-Kong_cheng_ji-nacta'],
    #  ['20170418TianHao', 'lsxp-Wo_zheng_zai-Kong_cheng_ji01-nacta'],
    #  ['20170418TianHao', 'lsxp-Wo_zheng_zai-Kong_cheng_ji02-nacta'],
    #  ['20170418TianHao', 'lsxp-Xi_ri_li-Zhu_lian_zhai-nacta'],
    #  ['20170424SunYuZhu', 'daeh-Yi_sha_shi-Suo_lin_nang-nacta'],
    #  ['20170424SunYuZhu', 'daxp-Dang_ri_li-Suo_lin_nang-nacta'],
    #  ['20170424SunYuZhu', 'daxp-Er_ting_de-Suo_lin_nang_take1-nacta'],
    #  ['20170424SunYuZhu', 'daxp-Er_ting_de-Suo_lin_nang_take2-nacta'],
    #  ['20170424SunYuZhu', 'daxp-Er_ting_de-Suo_lin_nang_take3-nacta'],
    #  ['20170424SunYuZhu', 'daxp-Zhe_cai_shi-Suo_lin_nang-nacta'],
    #  ['20170424SunYuZhu', 'daxp-Zhe_cai_shi-Suo_lin_nang_first_half-nacta'],
    #  ['20170425SunYuZhu', 'daeh-Wei_kai_yan-Dou_e_yuan-nacta'],
    #  ['20170425SunYuZhu', 'daxp-Chui_qiu_ting-Suo_lin_nang-nacta'],
    #  ['20170425SunYuZhu', 'daxp-Chui_qiu_ting-Suo_lin_nang_first_line-nacta'],
    #  ['20170506LiuHaiLin', 'daeh-Wang_chun_e-San_niang_jiao-ustb'],
    #  ['20170506LiuHaiLin', 'daeh-Wei_kai_yan-Dou_e_yuan-ustb'],
    #  ['20170506LiuHaiLin', 'daxp-Chun_qiu_ting-Suo_lin_nang-ustb'],
    #  ['20170506LiuHaiLin', 'daxp-Dang_ri_li-Suo_lin_nang-ustb'],
    #  ['20170506LiuHaiLin', 'daxp-Qiao_lou_shang-Huang_shan_lei-ustb'],
    #  ['20170506LiuHaiLin', 'daxp-Yi_sha_shi-Suo_lin_nang-ustb'],
    #  ['20170519LongTianMing', 'lseh-Tan_yang_jia-Hong_yang_dong-ustb'],
    #  ['20170519LongTianMing', 'lseh-Wei_guo_jia-Hong_yang_dong-ustb'],
    #  ['20170519LongTianMing', 'lseh-Yi_lun_ming-Zhuo_fang_cao-ustb'],
    #  ['20170519LongTianMing', 'lseh-Zi_na_ri-Hong_yang_dong-ustb'],
    #  ['20170519LongTianMing', 'lsxp-Ting_ta_yan-Zhuo_fang_cao-ustb'],
    #  ['20170519LongTianMing', 'lsxp-Wo_ben_shi-Kong_cheng_ji-ustb'],
    #  ['20170519LongTianMing', 'lsxp-Wo_zheng_zai-Kong_cheng_ji-ustb'],
    #  ['20170519XuJingWei', 'lseh-Jin_zhong_xiang-Shang_tian_tai-renmin'],
    #  ['20170519XuJingWei', 'lseh-Wei_guo_jia-Hong_yang_dong-renmin'],
    #  ['20170519XuJingWei', 'lseh-Wei_kai_yan-Rang_xu_zhou-renmin'],
    #  ['20170519XuJingWei', 'lsxp-Huai_nan_wang-Huai_he_ying-renmin'],
    #  ['20170519XuJingWei', 'lsxp-Jiang_shen_er-San_jia_dian-renmin'],
    #  ['20170519XuJingWei', 'lsxp-Wo_ben_shi-Kong_cheng_ji-renmin'],
    #  ['20170519XuJingWei', 'lsxp-Wo_zheng_zai-Kong_cheng_ji-renmin'],
    #  ['20170519XuJingWei', 'lsxp-Yi_ma_li-Wu_jia_po-renmin']]
    trainNacta = [['danAll', 'daeh-Yang_Yu_huan-Tai_zhen_wai_zhuan-lon'], ['danAll', 'dafeh-Bi_yun_tian-Xi_xiang_ji01-qm'],
     ['danAll', 'danbz-Bei_jiu_chan-Chun_gui_men01-qm'],
     ['danAll', 'danbz-Kan_dai_wang-Ba_wang_bie_ji01-qm'], ['danAll', 'daspd-Du_shou_kong-Wang_jiang_ting-upf'],
     ['danAll', 'daspd-Hai_dao_bing-Gui_fei_zui_jiu01-lon'], ['danAll', 'daspd-Hai_dao_bing-Gui_fei_zui_jiu02-qm'],
     ['danAll', 'daxp-Chun_qiu_ting-Suo_lin_nang01-qm'], ['danAll', 'daxp-Guan_Shi_yin-Tian_nv_san_hua-lon'],
     ['danAll', 'daxp-Jiao_Zhang_sheng-Hong_niang01-qm'], ['danAll', 'daxp-Jiao_Zhang_sheng-Hong_niang04-qm'],
     ['danAll', 'daxp-Meng_ting_de-Mu_Gui_ying_gua_shuai01-upf'],
     ['danAll', 'daxp-Meng_ting_de-Mu_Gui_ying_gua_shuai04-qm'],
     ['laosheng', 'lseh-Wei_guo_jia-Hong_yang_dong01-lon'],
      ['laosheng', 'lseh-Wo_ben_shi-Qiong_lin_yan-qm'],
     ['laosheng', 'lseh-Yi_lun_ming-Wen_zhao_guan-qm'], ['laosheng', 'lseh-Zi_na_ri-Hong_yang_dong-qm'],
      ['laosheng', 'lsxp-Guo_liao_yi-Wen_zhao_guan02-qm'],
     ['laosheng', 'lsxp-Huai_nan_wang-Huai_he_ying01-lon'], ['laosheng', 'lsxp-Huai_nan_wang-Huai_he_ying02-qm'],
     ['laosheng', 'lsxp-Jiang_shen_er-San_jia_dian01-1-upf'], ['laosheng', 'lsxp-Jiang_shen_er-San_jia_dian01-2-upf'],
     ['laosheng', 'lsxp-Jiang_shen_er-San_jia_dian02-qm'], ['laosheng', 'lsxp-Qian_bai_wan-Si_lang_tang_mu01-qm'],
     ['laosheng', 'lsxp-Quan_qian_sui-Gan_lu_si-qm'],
     ['laosheng', 'lsxp-Wo_zheng_zai-Kong_cheng_ji04-qm'], ['laosheng', 'lsxp-Xi_ri_you-Zhu_lian_zhai-qm']]
    testNacta = [['danAll', 'dagbz-Feng_xiao_xiao-Yang_men_nv_jiang-lon'],
                ['laosheng', 'lseh-Wei_guo_jia-Hong_yang_dong02-qm'],
                 ['laosheng', 'lseh-Tan_Yang_jia-Hong_yang_dong-qm'],
                 ['laosheng', 'lsxp-Shi_ye_shuo-Ding_jun_shan-qm'],
                 ['laosheng', 'lsxp-Wo_ben_shi-Kong_cheng_ji-qm'],
                 ['danAll', 'daxp-Zhe_cai_shi-Suo_lin_nang01-qm'],
                 ['danAll', 'daxp-Meng_ting_de-Mu_Gui_ying_gua_shuai02-qm'],
                 ['laosheng', 'lsxp-Guo_liao_yi-Wen_zhao_guan01-upf'],
                 ['laosheng', 'lsxp-Wo_zheng_zai-Kong_cheng_ji01-upf']]
    testNacta2017 = []

    return testNacta2017, testNacta, trainNacta2017, trainNacta

if __name__ == '__main__':
    # getTestTrainRecordings()
    # print getRecordingNames('TRAIN')

    # testNacta2017, testNacta, trainNacta2017, trainNacta = getTestTrainRecordingsMaleFemale()
    testNacta2017, testNacta, trainNacta2017, trainNacta = getTestTrainRecordingsArtistAlbumFilter()

    print(testNacta2017)
    print(testNacta)
    print(trainNacta2017)
    print(trainNacta)

    # testRiyaz, trainRiyaz = getTestTrainrecordingsRiyaz()
    # print(testRiyaz)
    # print(trainRiyaz)

    # getTestTrainRecordingsArtist()
