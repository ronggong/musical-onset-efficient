
"""
def getFeature(audio, d=True, nbf=False):

    '''
    MFCC of give audio interval [p[0],p[1]]
    :param audio:
    :param p:
    :return:
    '''

    winAnalysis = 'hann'

    # this MFCC is for pattern classification, which numberBands always be by default
    MFCC40 = ess.MFCC(sampleRate=fs,
                      highFrequencyBound=highFrequencyBound,
                      inputSize=framesize + 1)

    N = 2 * framesize  # padding 1 time framesize
    SPECTRUM = ess.Spectrum(size=N)
    WINDOW = ess.Windowing(type=winAnalysis, zeroPadding=N - framesize)

    mfcc   = []
    # audio_p = audio[p[0]*fs:p[1]*fs]
    for frame in ess.FrameGenerator(audio, frameSize=framesize, hopSize=hopsize):
        frame           = WINDOW(frame)
        mXFrame         = SPECTRUM(frame)
        bands,mfccFrame = MFCC40(mXFrame)
        # mfccFrame       = mfccFrame[1:]
        mfcc.append(mfccFrame)

    if d:
        mfcc            = np.array(mfcc).transpose()
        dmfcc           = Fdeltas(mfcc,w=5)
        ddmfcc          = Fdeltas(dmfcc,w=5)
        feature         = np.transpose(np.vstack((mfcc,dmfcc,ddmfcc)))
    else:
        feature         = np.array(mfcc)

    if not d and nbf:
        mfcc = np.array(mfcc).transpose()
        mfcc_out = np.array(mfcc, copy=True)
        for w_r in range(1,6):
            mfcc_right_shifted = Fprev_sub(mfcc, w=w_r)
            mfcc_left_shifted = Fprev_sub(mfcc, w=-w_r)
            mfcc_out = np.vstack((mfcc_out, mfcc_left_shifted, mfcc_right_shifted))
        feature = np.array(np.transpose(mfcc_out),dtype='float32')

    # print feature.shape

    return feature

def getMFCCBands1D(audio, nbf=False):

    '''
    mel bands feature [p[0],p[1]], this function only for pdnn acoustic model training
    output feature is a 1d vector
    it needs the array format float32
    :param audio:
    :param p:
    :param nbf: bool, if we need to neighbor frames
    :return:
    '''

    winAnalysis = 'hann'

    MFCC80 = ess.MFCC(sampleRate=fs,
                      highFrequencyBound=highFrequencyBound,
                      inputSize=framesize + 1,
                      numberBands=80)

    N = 2 * framesize  # padding 1 time framesize
    SPECTRUM = ess.Spectrum(size=N)
    WINDOW = ess.Windowing(type=winAnalysis, zeroPadding=N - framesize)

    mfcc   = []
    # audio_p = audio[p[0]*fs:p[1]*fs]
    for frame in ess.FrameGenerator(audio, frameSize=framesize, hopSize=hopsize):
        frame           = WINDOW(frame)
        mXFrame         = SPECTRUM(frame)
        bands,mfccFrame = MFCC80(mXFrame)
        mfcc.append(bands)

    if nbf:
        mfcc = np.array(mfcc).transpose()
        mfcc_right_shifted_1 = Fprev_sub(mfcc, w=1)
        mfcc_left_shifted_1 = Fprev_sub(mfcc, w=-1)
        mfcc_right_shifted_2 = Fprev_sub(mfcc, w=2)
        mfcc_left_shifted_2 = Fprev_sub(mfcc, w=-2)
        feature = np.transpose(np.vstack((mfcc,
                                          mfcc_right_shifted_1,
                                          mfcc_left_shifted_1,
                                          mfcc_right_shifted_2,
                                          mfcc_left_shifted_2)))
    else:
        feature = mfcc

    # the mel bands features
    feature = np.array(feature,dtype='float32')

    return feature


def getMFCCBands2D(audio, framesize_t, hopsize_t, fs, nbf=False, nlen=10):

    '''
    mel bands feature [p[0],p[1]]
    output feature for each time stamp is a 2D matrix
    it needs the array format float32
    :param audio:
    :param p:
    :param nbf: bool, if we need neighbor frames
    :return:
    '''

    winAnalysis = 'hann'

    highFrequencyBound = fs / 2 if fs / 2 < 11000 else 11000

    framesize = int(round(framesize_t * fs))
    hopsize = int(round(hopsize_t * fs))

    MFCC80 = ess.MFCC(sampleRate=fs,
                      highFrequencyBound=highFrequencyBound,
                      inputSize=framesize + 1,
                      numberBands=80)

    N = 2 * framesize  # padding 1 time framesize
    SPECTRUM = ess.Spectrum(size=N)
    WINDOW = ess.Windowing(type=winAnalysis, zeroPadding=N - framesize)

    mfcc   = []
    # audio_p = audio[p[0]*fs:p[1]*fs]
    for frame in ess.FrameGenerator(audio, frameSize=framesize, hopSize=hopsize):
        frame           = WINDOW(frame)
        mXFrame         = SPECTRUM(frame)
        bands,mfccFrame = MFCC80(mXFrame)
        mfcc.append(bands)

    if nbf:
        feature = _nbf_2D(mfcc, nlen)
    else:
        feature = mfcc
    # the mel bands features
    feature = np.array(feature,dtype='float32')

    return feature


def featureExtraction(audio_monoloader, scaler, framesize_t, hopsize_t, fs, dmfcc=False, nbf=False, feature_type='mfccBands2D'):
    '''
    extract mfcc features
    :param audio_monoloader:
    :param scaler:
    :param dmfcc:
    :param nbf:
    :param feature_type:
    :return:
    '''
    if feature_type == 'mfccBands2D':
        mfcc = getMFCCBands2D(audio_monoloader, framesize_t, hopsize_t, fs, nbf=nbf, nlen=varin['nlen'])
        mfcc = np.log(100000 * mfcc + 1)

        mfcc = np.array(mfcc, dtype='float32')
        mfcc_scaled = scaler.transform(mfcc)
        mfcc_reshaped = featureReshape(mfcc_scaled)
    else:
        print(feature_type + ' is not exist.')
        raise
    return mfcc, mfcc_reshaped


def getMBE(audio):
    '''
    mel band energy feature
    :param audio:
    :return:
    '''

    winAnalysis = 'hann'

    # this MFCC is for pattern classification, which numberBands always be by default
    MFCC40 = ess.MFCC(sampleRate=fs,
                      highFrequencyBound=highFrequencyBound,
                      inputSize=framesize + 1)

    N = 2 * framesize  # padding 1 time framesize
    SPECTRUM = ess.Spectrum(size=N)
    WINDOW = ess.Windowing(type=winAnalysis, zeroPadding=N - framesize)

    mfccBands = []
    for frame in ess.FrameGenerator(audio, frameSize=framesize, hopSize=hopsize):

        frame           = WINDOW(frame)
        mXFrame         = SPECTRUM(frame)
        bands,mfccFrame = MFCC40(mXFrame)
        mfccBands.append(bands)
    feature         = np.array(mfccBands)
    return feature


def dumpFeaturePhoneme(full_path_recordings,
                       full_path_textgrids,
                       syllableTierName,
                       phonemeTierName,
                       feature_type='mfcc',
                       dmfcc=True,
                       nbf=False):
    '''
    dump the MFCC for each phoneme
    :param recordings:
    :return:
    '''

    ##-- dictionary feature
    dic_pho_feature = {}

    for _,pho in enumerate(set(dic_pho_map.values())):
        dic_pho_feature[pho] = np.array([])

    for ii_rec, recording in enumerate(full_path_recordings):

        lineList = textGrid2WordList(full_path_textgrids[ii_rec], whichTier=syllableTierName)
        utteranceList = textGrid2WordList(full_path_textgrids[ii_rec], whichTier=phonemeTierName)

        # parse lines of groundtruth
        nestedPhonemeLists, _, _ = wordListsParseByLines(lineList, utteranceList)

        # audio
        wav_full_filename   = recording
        audio               = ess.MonoLoader(downmix = 'left', filename = wav_full_filename, sampleRate = fs)()

        if feature_type == 'mfcc':
            # MFCC feature
            mfcc = getFeature(audio, d=dmfcc, nbf=nbf)
        elif feature_type == 'mfccBands1D':
            mfcc = getMFCCBands1D(audio, nbf=nbf)
            mfcc = np.log(100000*mfcc+1)
        elif feature_type == 'mfccBands2D':
            mfcc = getMFCCBands2D(audio, framesize_t, hopsize_t, fs, nbf=nbf, nlen=varin['nlen'])
            mfcc = np.log(100000*mfcc+1)
        else:
            print(feature_type+' is not exist.')
            raise

        for ii,pho in enumerate(nestedPhonemeLists):
            print 'calculating ', recording, ' and phoneme ', str(ii), ' of ', str(len(nestedPhonemeLists))
            for p in pho[1]:
                # map from annotated xsampa to readable notation
                key = dic_pho_map[p[2]]

                sf = int(round(p[0]*fs/float(hopsize))) # starting frame
                ef = int(round(p[1]*fs/float(hopsize))) # ending frame

                mfcc_p = mfcc[sf:ef,:]  # phoneme syllable

                if not len(dic_pho_feature[key]):
                    dic_pho_feature[key] = mfcc_p
                else:
                    dic_pho_feature[key] = np.vstack((dic_pho_feature[key],mfcc_p))

    return dic_pho_feature


def dumpFeatureOnsetHelper(lab,
                           wav_path,
                           textgrid_path,
                           score_path,
                           artist_name,
                           recording_name,
                           feature_type,
                           nbf):
    if not lab:
        groundtruth_textgrid_file = os.path.join(textgrid_path, artist_name, recording_name + '.TextGrid')
        print(groundtruth_textgrid_file)
        wav_file = os.path.join(wav_path, artist_name, recording_name + '.wav')
    else:
        groundtruth_textgrid_file = os.path.join(textgrid_path, artist_name, recording_name + '.lab')
        wav_file = os.path.join(wav_path, artist_name, recording_name + '.mp3')

    if '2017' in artist_name:
        score_file = os.path.join(score_path, artist_name, recording_name + '.csv')
    else:
        score_file = os.path.join(score_path, artist_name, recording_name + '.csv')

    # if not os.path.isfile(score_file):
    #     print 'Score not found: ' + score_file
    #     continue

    if not lab:
        lineList = textGrid2WordList(groundtruth_textgrid_file, whichTier='line')
        utteranceList = textGrid2WordList(groundtruth_textgrid_file, whichTier='dianSilence')

        # parse lines of groundtruth
        nestedUtteranceLists, numLines, numUtterances = wordListsParseByLines(lineList, utteranceList)
    else:
        nestedUtteranceLists = [lab2WordList(groundtruth_textgrid_file, label=True)]

    # parse score
    _, utterance_durations, bpm = csvDurationScoreParser(score_file)

    # load audio
    fs = 44100

    if feature_type != 'madmom':
        if not lab:
            audio = ess.MonoLoader(downmix='left', filename=wav_file, sampleRate=fs)()
        else:
            audio, fs, nc, md5, br, codec = ess.AudioLoader(filename=wav_file)()
            audio = audio[:, 0]  # take the left channel

    if feature_type == 'mfccBands2D':
        mfcc = getMFCCBands2D(audio, framesize_t, hopsize_t, fs, nbf=nbf, nlen=varin['nlen'])
        mfcc = np.log(100000 * mfcc + 1)
    elif feature_type == 'madmom':
        mfcc = getMFCCBands2DMadmom(wav_file, fs, hopsize_t, channel=1)
    else:
        print(feature_type + ' is not exist.')
        raise
    return nestedUtteranceLists, utterance_durations, bpm, mfcc


def dumpFeatureBatchOnsetRiyaz():
    '''
    dump feature for Riyaz dataset
    :return:
    '''
    testRiyaz, trainRiyaz = getTestTrainrecordingsRiyaz()

    mfcc_p, \
    mfcc_n, \
    sample_weights_p, \
    sample_weights_n \
        = dumpFeatureOnset(wav_path=riyaz_mp3_path,
                           textgrid_path=riyaz_groundtruthlab_path,
                           score_path=riyaz_score_path,
                           recordings=trainRiyaz,
                           feature_type='mfccBands2D',
                           dmfcc=False,
                           nbf=True,
                           lab=True)

    sample_weights = np.concatenate((sample_weights_p, sample_weights_n))

    feature_all, label_all, scaler = featureLabelOnset(mfcc_p, mfcc_n)

    print(mfcc_p.shape, mfcc_n.shape, sample_weights_p.shape, sample_weights_n.shape)

    pickle.dump(scaler, open('../cnnModels/scaler_syllable_mfccBands2D_riyaz'+str(varin['nlen'])+'.pkl', 'wb'))

    feature_all = featureReshape(feature_all, nlen=varin['nlen'])

    print(feature_all.shape)

    h5f = h5py.File(join(riyaz_feature_data_path, 'feature_all_riyaz'+str(varin['nlen'])+'.h5'), 'w')
    h5f.create_dataset('feature_all', data=feature_all)
    h5f.close()

    cPickle.dump(label_all,
                 gzip.open('../trainingData/labels_train_set_all_syllableSeg_mfccBands2D_riyaz'+str(varin['nlen'])+'.pickle.gz', 'wb'),
                 cPickle.HIGHEST_PROTOCOL)

    cPickle.dump(sample_weights,
                 gzip.open('../trainingData/sample_weights_syllableSeg_mfccBands2D_riyaz'+str(varin['nlen'])+'.pickle.gz', 'wb'),
                 cPickle.HIGHEST_PROTOCOL)
"""