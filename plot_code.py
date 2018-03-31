# import matplotlib
# matplotlib.use('Tkagg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_jingju(nested_syllable_lists,
                i_line,
                mfcc_line,
                hopsize_t,
                obs_i,
                i_boundary,
                duration_score):
    # print(line_list)
    nested_ul = nested_syllable_lists[i_line][1]
    # print(nested_ul)
    ground_truth_onset = [l[0] - nested_ul[0][0] for l in nested_ul]

    # nested_ul = line_list[0]
    # ground_truth_onset = [l[0]-line[0] for l in nested_ul]
    # groundtruth_syllables = [l[2] for l in nested_ul]

    # plot Error analysis figures
    plt.figure(figsize=(16, 6))
    # class weight
    ax1 = plt.subplot(3, 1, 1)
    y = np.arange(0, 80)
    x = np.arange(0, mfcc_line.shape[0]) * hopsize_t
    plt.pcolormesh(x, y, np.transpose(mfcc_line[:, 80 * 7:80 * 8]))
    for i_gs, gs in enumerate(ground_truth_onset):
        plt.axvline(gs, color='r', linewidth=2)

    ax1.set_ylabel('Mel bands', fontsize=12)
    ax1.get_xaxis().set_visible(False)
    ax1.axis('tight')

    # detected onsets
    ax2 = plt.subplot(312, sharex=ax1)
    plt.plot(np.arange(0, len(obs_i)) * hopsize_t, obs_i)
    for i_ib in range(len(i_boundary) - 1):
        plt.axvline(i_boundary[i_ib] * hopsize_t, color='r', linewidth=2)
    ax2.set_ylabel('ODF', fontsize=12)
    ax2.axis('tight')

    # plot the score durations
    ax3 = plt.subplot(313, sharex=ax1)
    time_start = 0
    for ii_ds, ds in enumerate(duration_score):
        ax3.add_patch(
            patches.Rectangle(
                (time_start, ii_ds),  # (x,y)
                ds,  # width
                1,  # height
            ))
        time_start += ds
    ax3.set_ylim((0, len(duration_score)))
    ax3.set_ylabel('Score duration', fontsize=12)
    ax3.axis('tight')

    plt.xlabel('Time (s)')
    plt.show()


def plot_schluter(mfcc,
                  obs_i,
                  hopsize_t,
                  groundtruth_onset,
                  detected_onsets):

    plt.figure(figsize=(16, 6))

    ax1 = plt.subplot(2, 1, 1)
    y = np.arange(0, 80)
    x = np.arange(0, mfcc.shape[0]) * hopsize_t
    plt.pcolormesh(x, y, np.transpose(mfcc[:, 80 * 10:80 * 11]))
    for i_gs, gs in enumerate(groundtruth_onset):
        plt.axvline(gs, color='r', linewidth=2)

    ax1.set_ylabel('Mel bands', fontsize=12)
    ax1.get_xaxis().set_visible(False)
    ax1.axis('tight')

    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(np.arange(0, len(obs_i)) * hopsize_t, obs_i)
    for i_do, do in enumerate(detected_onsets):
        plt.axvline(do, color='r', linewidth=2)

    ax2.set_ylabel('ODF', fontsize=12)
    ax2.axis('tight')
    plt.xlabel('time (s)')
    plt.show()


def plot_jingju_odf_colab(log_mel,
                          hopsize_t,
                          odf_baseline,
                          odf_relu_dense,
                          odf_no_dense,
                          odf_temporal,
                          odf_bidi_lstms_100,
                          odf_bidi_lstms_200,
                          odf_bidi_lstms_400,
                          odf_9_layers_cnn,
                          odf_5_layers_cnn,
                          odf_pretrained,
                          odf_retrained,
                          odf_feature_extractor_a,
                          odf_feature_extractor_b):

    # plot Error analysis figures
    plt.figure(figsize=(16, 26))
    # class weight
    ax1 = plt.subplot(14, 1, 1)
    y = np.arange(0, 80)
    x = np.arange(0, log_mel.shape[0]) * hopsize_t
    plt.pcolormesh(x, y, np.transpose(log_mel[:, 80 * 7:80 * 8]))
    ax1.set_ylabel('Mel bands', fontsize=12)
    ax1.get_xaxis().set_visible(False)
    ax1.axis('tight')

    # detected onsets
    ax2 = plt.subplot(14, 1, 2, sharex=ax1)
    plt.plot(np.arange(0, len(odf_baseline)) * hopsize_t, odf_baseline)
    ax2.set_ylabel('ODF Baseline', fontsize=12)
    ax2.axis('tight')

    ax3 = plt.subplot(14, 1, 3, sharex=ax1)
    plt.plot(np.arange(0, len(odf_relu_dense)) * hopsize_t, odf_relu_dense)
    ax3.set_ylabel('ODF ReLU dense', fontsize=12)
    ax3.axis('tight')

    ax4 = plt.subplot(14, 1, 4, sharex=ax1)
    plt.plot(np.arange(0, len(odf_no_dense)) * hopsize_t, odf_no_dense)
    ax4.set_ylabel('ODF No dense', fontsize=12)
    ax4.axis('tight')

    ax5 = plt.subplot(14, 1, 5, sharex=ax1)
    plt.plot(np.arange(0, len(odf_temporal)) * hopsize_t, odf_temporal)
    ax5.set_ylabel('ODF Temporal', fontsize=12)
    ax5.axis('tight')

    ax6 = plt.subplot(14, 1, 6, sharex=ax1)
    plt.plot(np.arange(0, len(odf_bidi_lstms_100)) * hopsize_t, odf_bidi_lstms_100)
    ax6.set_ylabel('ODF Bidi\nLSTMs 100', fontsize=12)
    ax6.axis('tight')

    ax7 = plt.subplot(14, 1, 7, sharex=ax1)
    plt.plot(np.arange(0, len(odf_bidi_lstms_200)) * hopsize_t, odf_bidi_lstms_200)
    ax7.set_ylabel('ODF Bidi\nLSTMs 200', fontsize=12)
    ax7.axis('tight')

    ax8 = plt.subplot(14, 1, 8, sharex=ax1)
    plt.plot(np.arange(0, len(odf_bidi_lstms_400)) * hopsize_t, odf_bidi_lstms_400)
    ax8.set_ylabel('ODF Bidi\nLSTMs 400', fontsize=12)
    ax8.axis('tight')

    ax9 = plt.subplot(14, 1, 9, sharex=ax1)
    plt.plot(np.arange(0, len(odf_9_layers_cnn)) * hopsize_t, odf_9_layers_cnn)
    ax9.set_ylabel('ODF \n9 layers CNN', fontsize=12)
    ax9.axis('tight')

    ax10 = plt.subplot(14, 1, 10, sharex=ax1)
    plt.plot(np.arange(0, len(odf_5_layers_cnn)) * hopsize_t, odf_5_layers_cnn)
    ax10.set_ylabel('ODF \n5 layers CNN', fontsize=12)
    ax10.axis('tight')

    ax11 = plt.subplot(14, 1, 11, sharex=ax1)
    plt.plot(np.arange(0, len(odf_pretrained)) * hopsize_t, odf_pretrained)
    ax11.set_ylabel('ODF pretrained', fontsize=12)
    ax11.axis('tight')

    ax12 = plt.subplot(14, 1, 12, sharex=ax1)
    plt.plot(np.arange(0, len(odf_retrained)) * hopsize_t, odf_retrained)
    ax12.set_ylabel('ODF retrained', fontsize=12)
    ax12.axis('tight')

    ax13 = plt.subplot(14, 1, 13, sharex=ax1)
    plt.plot(np.arange(0, len(odf_feature_extractor_a)) * hopsize_t, odf_feature_extractor_a)
    ax13.set_ylabel('ODF feature\nextractor a', fontsize=12)
    ax13.axis('tight')

    ax14 = plt.subplot(14, 1, 14, sharex=ax1)
    plt.plot(np.arange(0, len(odf_feature_extractor_b)) * hopsize_t, odf_feature_extractor_b)
    ax14.set_ylabel('ODF feature\nextractor b', fontsize=12)
    ax14.axis('tight')

    plt.xlabel('Time (s)')
    plt.show()