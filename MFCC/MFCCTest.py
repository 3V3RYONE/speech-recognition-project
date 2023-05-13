#!/usr/bin/env python
from MFCC import mfcc
from MFCC import delta
from MFCC import logfbank
import scipy.io.wavfile as wav
import numpy as np
from scipy.spatial.distance import euclidean, cosine, cityblock, chebyshev, canberra, braycurtis, mahalanobis, minkowski
from scipy.signal import resample

(rate,sig) = wav.read("../sounds/Beleswar.wav")
mfcc_feat = mfcc(sig, rate)
d_mfcc_feat = delta(mfcc_feat, 2)
fbank_feat = logfbank(sig, rate)

(rate1,sig1) = wav.read("../sounds/Jay.wav")
mfcc_feat1 = mfcc(sig1, rate1)
d_mfcc_feat1 = delta(mfcc_feat1, 2)
fbank_feat1 = logfbank(sig1, rate1)

mfcc_1 = np.array(fbank_feat).flatten()
mfcc_2 = np.array(fbank_feat1).flatten()

print(mfcc_1)
print(mfcc_2)

if mfcc_1.shape[0] > mfcc_2.shape[0]:
    mfcc_1 = resample(mfcc_1, mfcc_2.shape[0])
elif mfcc_2.shape[0] > mfcc_1.shape[0]:
    mfcc_2 = resample(mfcc_2, mfcc_1.shape[0])

distances = [
    ('Euclidean', euclidean(mfcc_1, mfcc_2)),
    ('Cosine', cosine(mfcc_1, mfcc_2)),
    ('Manhattan', cityblock(mfcc_1, mfcc_2)),
    ('Chebyshev', chebyshev(mfcc_1, mfcc_2)),
    ('Canberra', canberra(mfcc_1, mfcc_2)),
    ('Braycurtis', braycurtis(mfcc_1, mfcc_2)),
    ('Minkowski', minkowski(mfcc_1, mfcc_2, p=2)),
]

for d in distances:
    print(f"{d[0]} distance: {d[1]}")

best_score = float('inf')
best_algo = ''
for d in distances:
    print(f"{d[0]} distance: {d[1]}")
    similarity = (1 - d[1]) * 100
    print(f"{d[0]} similarity: {similarity}")
    if d[1] < best_score:
        best_score = d[1]
        best_algo = d[0]
print(f"\nBest algorithm: {best_algo}, with a score of {best_score} and a similarity of {(1 - best_score) * 100}%")
