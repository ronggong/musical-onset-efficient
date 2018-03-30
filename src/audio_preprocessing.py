import numpy as np
from Fprev_sub import Fprev_sub
from madmom.processors import SequentialProcessor, ParallelProcessor

EPSILON = np.spacing(1)

def _nbf_2D(mfcc, nlen):
    mfcc = np.array(mfcc).transpose()
    mfcc_out = np.array(mfcc, copy=True)
    for ii in range(1, nlen + 1):
        mfcc_right_shift = Fprev_sub(mfcc, w=ii)
        mfcc_left_shift = Fprev_sub(mfcc, w=-ii)
        mfcc_out = np.vstack((mfcc_right_shift, mfcc_out, mfcc_left_shift))
    feature = mfcc_out.transpose()
    return feature

class MadmomMelbankProcessor(SequentialProcessor):


    def __init__(self, fs, hopsize_t):
        from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
        from madmom.audio.stft import ShortTimeFourierTransformProcessor
        from madmom.audio.filters import MelFilterbank
        from madmom.audio.spectrogram import (FilteredSpectrogramProcessor,
                                              LogarithmicSpectrogramProcessor)
        # from madmom.features.onsets import _cnn_onset_processor_pad

        # define pre-processing chain
        sig = SignalProcessor(num_channels=1, sample_rate=fs)
        # process the multi-resolution spec in parallel
        # multi = ParallelProcessor([])
        # for frame_size in [2048, 1024, 4096]:
        frames = FramedSignalProcessor(frame_size=2048, hopsize=int(fs*hopsize_t))
        stft = ShortTimeFourierTransformProcessor()  # caching FFT window
        filt = FilteredSpectrogramProcessor(
            filterbank=MelFilterbank, num_bands=80, fmin=27.5, fmax=16000,
            norm_filters=True, unique_filters=False)
        spec = LogarithmicSpectrogramProcessor(log=np.log, add=EPSILON)

        # process each frame size with spec and diff sequentially
        # multi.append())
        single = SequentialProcessor([frames, stft, filt, spec])

        # stack the features (in depth) and pad at beginning and end
        # stack = np.dstack
        # pad = _cnn_onset_processor_pad

        # pre-processes everything sequentially
        pre_processor = SequentialProcessor([sig, single])

        # instantiate a SequentialProcessor
        super(MadmomMelbankProcessor, self).__init__([pre_processor])


class MadmomMelbank3ChannelsProcessor(SequentialProcessor):


    def __init__(self, fs, hopsize_t):
        from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
        from madmom.audio.stft import ShortTimeFourierTransformProcessor
        from madmom.audio.filters import MelFilterbank
        from madmom.audio.spectrogram import (FilteredSpectrogramProcessor,
                                              LogarithmicSpectrogramProcessor)
        # from madmom.features.onsets import _cnn_onset_processor_pad

        # define pre-processing chain
        sig = SignalProcessor(num_channels=1, sample_rate=fs)
        # process the multi-resolution spec in parallel
        multi = ParallelProcessor([])
        for frame_size in [2048, 1024, 4096]:
            frames = FramedSignalProcessor(frame_size=frame_size, fps=100)
            stft = ShortTimeFourierTransformProcessor()  # caching FFT window
            filt = FilteredSpectrogramProcessor(
                filterbank=MelFilterbank, num_bands=80, fmin=27.5, fmax=16000,
                norm_filters=True, unique_filters=False)
            spec = LogarithmicSpectrogramProcessor(log=np.log, add=EPSILON)
            # process each frame size with spec and diff sequentially
            multi.append(SequentialProcessor([frames, stft, filt, spec]))
        # stack the features (in depth) and pad at beginning and end
        stack = np.dstack
        # pad = _cnn_onset_processor_pad
        # pre-processes everything sequentially
        pre_processor = SequentialProcessor([sig, multi, stack])
        # instantiate a SequentialProcessor
        super(MadmomMelbank3ChannelsProcessor, self).__init__([pre_processor])


def getMFCCBands2DMadmom(audio_fn, fs, hopsize_t, channel):
    if channel == 1:
        madmomMelbankProc = MadmomMelbankProcessor(fs, hopsize_t)
    else:
        madmomMelbankProc = MadmomMelbank3ChannelsProcessor(fs, hopsize_t)

    mfcc = madmomMelbankProc(audio_fn)

    if channel == 1:
        mfcc = _nbf_2D(mfcc, 7)
    else:
        mfcc_conc = []
        for ii in range(3):
            mfcc_conc.append(_nbf_2D(mfcc[:,:,ii], 7))
        mfcc = np.stack(mfcc_conc, axis=2)
    return mfcc