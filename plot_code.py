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