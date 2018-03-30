# parameters of ODF onset detection function

# ODF method: 'jordi': Pons' CNN, 'jan': Schluter' CNN, 'jan_chan3'
mth_ODF = 'jan'

# layer2 node number: 20 or 32
layer2 = 20

# late fusion: Bool
fusion = False

# filter shape: 'temporal' or 'timbral' filter shape in Pons' CNN
filter_shape = 'temporal'


# other params
fs = 44100
framesize_t = 0.025     # in second
hopsize_t = 0.010

framesize = int(round(framesize_t*fs))
hopsize = int(round(hopsize_t*fs))

# MFCC params
highFrequencyBound = fs/2 if fs/2 < 11000 else 11000

varin = {}

# parameters of viterbi
varin['delta_mode'] = 'proportion'

varin['delta'] = 0.35

varin['plot'] = False

varin['decoding'] = 'peakPicking'

varin['obs'] = 'tocal'

varin['corrected_score_duration'] = False

varin['dataset'] = 'artist_filter'

varin['sample_weighting'] = 'simpleWeighting'

varin['overlap'] = False

varin['bidi'] = False

# score synthesis parameters
framesize_melodicSimilarity = 2048
hopsize_melodicSimilarity = 1024
synthesizeLength = 5  # in second
sample_number_total = int(round(synthesizeLength * (fs / float(hopsize_melodicSimilarity))))