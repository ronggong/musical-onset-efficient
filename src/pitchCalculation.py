import essentia.standard as ess

def pitchCalculation(audio, start_end_samples, frameSize, sampleRate, maxFrequency):

    PITCHYIN = ess.PitchYin(frameSize=frameSize, sampleRate=sampleRate, maxFrequency=maxFrequency)

    pitch, pitchConfidence = PITCHYIN(audio[start_end_samples[0]:start_end_samples[1]])
    return pitch, pitchConfidence