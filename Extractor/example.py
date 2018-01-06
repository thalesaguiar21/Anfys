#!/usr/bin/env python

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import numpy
numpy.set_printoptions(threshold='nan')

(rate, sig) = wav.read("OSR_us_000_0010_8k.wav")
print('Computing MFCC features for {0}, SIG: {1}', rate, len(sig))
mfcc_feat = mfcc(sig, rate)
print('Done!')
#d_mfcc_feat = delta(mfcc_feat, 2)
#fbank_feat = logfbank(sig, rate)


mfcc_file = open('mfcc_result.txt', 'w')
print('Writing to file')

for feat in mfcc_feat:
    for freq in feat:
        mfcc_file.write(str(freq) + '\t')
    mfcc_file.write('\n')
mfcc_file.close()
