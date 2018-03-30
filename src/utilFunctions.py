import numpy as np
import os

def featureReshape(feature, nlen=10):
    """
    reshape mfccBands feature into n_sample * n_row * n_col
    :param feature:
    :return:
    """

    n_sample = feature.shape[0]
    n_row = 80
    n_col = nlen*2+1

    feature_reshaped = np.zeros((n_sample,n_row,n_col),dtype='float32')
    # print("reshaping feature...")
    for ii in range(n_sample):
        # print ii
        feature_frame = np.zeros((n_row,n_col),dtype='float32')
        for jj in range(n_col):
            feature_frame[:,jj] = feature[ii][n_row*jj:n_row*(jj+1)]
        feature_reshaped[ii,:,:] = feature_frame
    return feature_reshaped


def featureDereshape(feature, nlen=10):
    """
    de reshape the feature
    :param feature:
    :param nlen:
    :return:
    """
    n_sample = feature.shape[0]
    n_row = 80
    n_col = nlen * 2 + 1

    feature_dereshape = np.zeros((n_sample, n_row*n_col), dtype='float32')
    for ii in range(n_sample):
        for jj in range(n_col):
            feature_dereshape[ii][n_row*jj:n_row*(jj+1)] = feature[ii][:,jj]
    return feature_dereshape

def getRecordings(wav_path):
    recordings      = []
    for root, subFolders, files in os.walk(wav_path):
            for f in files:
                file_prefix, file_extension = os.path.splitext(f)
                if file_prefix != '.DS_Store' and file_prefix != '_DS_Store':
                    recordings.append(file_prefix)

    return recordings


def getOnsetFunction(observations, model, method):
    """
    Load CNN model to calculate ODF
    :param observations:
    :return:
    """

    if 'jordi' in method:
        observations = [observations, observations, observations, observations, observations, observations]
    elif 'feature_extractor' in method:
        observations = [observations, observations]

    obs = model.predict(observations, batch_size=128, verbose=2)

    return obs

def trackOnsetPosByPath(path, idx_syllable_start_state):
    idx_onset = []
    for ii in range(len(path)-1):
        if path[ii+1] != path[ii] and path[ii+1] in idx_syllable_start_state:
            idx_onset.append(ii)
    return idx_onset

def late_fusion_calc(obs_0, obs_1, mth=0, coef=0.5):
    """
    Late fusion methods
    :param obs_0:
    :param obs_1:
    :param mth: 0-addition 1-addition with linear norm 2-exponential weighting mulitplication with linear norm
    3-multiplication 4- multiplication with linear norm
    :param coef: weighting coef for multiplication
    :return:
    """
    assert len(obs_0) == len(obs_1)

    obs_out = []

    if mth==1 or mth==2 or mth==4:
        from sklearn.preprocessing import MinMaxScaler
        min_max_scaler = MinMaxScaler()
        obs_0 = obs_0.reshape((len(obs_0),1))
        obs_1 = obs_1.reshape((len(obs_1),1))
        # print(obs_0.shape, obs_1.shape)
        obs_0 = min_max_scaler.fit_transform(obs_0)
        obs_1 = min_max_scaler.fit_transform(obs_1)

    if mth == 0 or mth == 1:
        # addition
        obs_out = np.add(obs_0, obs_1)/2
    elif mth == 2:
        # multiplication with exponential weighting
        obs_out = np.multiply(np.power(obs_0, coef), np.power(obs_1, 1-coef))
    elif mth == 3 or mth == 4:
        # multiplication
        obs_out = np.multiply(obs_0, obs_1)

    return obs_out

def late_fusion_calc_3(obs_0, obs_1, obs_2, mth=2, coef=0.33333333):
    """
    Late fusion methods
    :param obs_0:
    :param obs_1:
    :param mth: 0-addition 1-addition with linear norm 2-exponential weighting mulitplication with linear norm
    3-multiplication 4- multiplication with linear norm
    :param coef: weighting coef for multiplication
    :return:
    """
    assert len(obs_0) == len(obs_1)

    obs_out = []

    if mth==2:
        from sklearn.preprocessing import MinMaxScaler
        min_max_scaler = MinMaxScaler()
        obs_0 = obs_0.reshape((len(obs_0),1))
        obs_1 = obs_1.reshape((len(obs_1),1))
        obs_2 = obs_2.reshape((len(obs_2),1))

        # print(obs_0.shape, obs_1.shape)
        obs_0 = min_max_scaler.fit_transform(obs_0)
        obs_1 = min_max_scaler.fit_transform(obs_1)
        obs_2 = min_max_scaler.fit_transform(obs_2)


    if mth == 2:
        # multiplication with exponential weighting
        obs_out = np.multiply(np.power(obs_0, coef), np.power(obs_1, coef))
        obs_out = np.multiply(obs_out, np.power(obs_2, 1-coef))
    else:
        raise ValueError

    return obs_out


def append_or_write(eval_result_file_name):
    if os.path.exists(eval_result_file_name):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not
    return append_write


def hz2cents(pitchInHz, tonic=261.626):
    cents = 1200*np.log2(1.0*pitchInHz/tonic)
    return cents


def pitchtrackInterp(pitchInCents, sample_number_total):
    """
    Interpolate the pitch track into sample_number_total points
    :param pitchInCents:
    :param sample_number_total:
    :return:
    """
    x = np.linspace(0, 100, len(pitchInCents))
    xvals = np.linspace(0, 100, sample_number_total)
    pitchInCents_interp = np.interp(xvals, x, pitchInCents)
    return pitchInCents_interp.tolist()


def stringDist(str0, str1):
    '''
    utf-8 format string
    :param str0:
    :param str1:
    :return:
    '''

    intersection = [val for val in str0 if val in str1]

    dis = len(intersection)/float(max(len(str0), len(str1)))

    return dis


def smooth_obs(obs):
    """using moving average hanning window for smoothing"""
    hann = np.hanning(5)
    hann /= np.sum(hann)
    obs = np.convolve(hann, obs, mode='same')
    return obs