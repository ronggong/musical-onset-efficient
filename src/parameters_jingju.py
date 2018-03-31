# other params
fs = 44100
framesize_t = 0.025     # in second
hopsize_t = 0.010

framesize = int(round(framesize_t*fs))
hopsize = int(round(hopsize_t*fs))

highFrequencyBound = fs/2 if fs/2 < 11000 else 11000

# CRNN batch size
batch_size = 64

varin = {}

# parameters of viterbi
varin['delta_mode'] = 'proportion'

varin['delta'] = 0.35

varin['plot'] = True

varin['decoding'] = 'peakPicking'

varin['obs'] = 'tocal'

varin['corrected_score_duration'] = False

varin['dataset'] = 'artist_filter'

varin['sample_weighting'] = 'simpleWeighting'

varin['overlap'] = True

varin['bidi'] = True

# varin['architecture'] = 'jan_less_deep'