
def getTestTrainRecordingsArtistAlbumFilter():

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


def getTestTrainRecordingsNactaISMIR():
    """
    coherent to ismir 2017 paper datasplit
    :return:
    """

    trainNacta2017 = []

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

    _, _, trainNacta2017, trainNacta = getTestTrainRecordingsArtistAlbumFilter()
    testNacta2017, testNacta = getTestRecordingsScoreDurCorrectionArtistAlbumFilter()

    print(len(trainNacta2017))
    print(len(trainNacta))
    print(len(testNacta2017))
    print(len(testNacta))
    print(len(trainNacta2017) + len(trainNacta) + len(testNacta2017) + len(testNacta))
