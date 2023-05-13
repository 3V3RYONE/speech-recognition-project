import librosa
import numpy as np
from scipy.signal import resample
# Load the audio signals
audio1, sr1 = librosa.load('../sounds/Beleswar.wav')
audio2, sr2 = librosa.load('../sounds/Jay.wav')

# Extract the pitch features from the audio signals
pitch1, _ = librosa.piptrack(y=audio1, sr=sr1)
pitch2, _ = librosa.piptrack(y=audio2, sr=sr2)

# Take the mean pitch value for each frame
mean_pitch1 = np.mean(pitch1, axis=0)
mean_pitch2 = np.mean(pitch2, axis=0)

if mean_pitch1.shape[0] > mean_pitch2.shape[0]:
    mean_pitch1 = resample(mean_pitch1, mean_pitch2.shape[0])
elif mean_pitch2.shape[0] > mean_pitch1.shape[0]:
    mean_pitch2 = resample(mean_pitch2, mean_pitch1.shape[0])

# Calculate the Euclidean distance between the pitch features
euclidean_distance = np.linalg.norm(mean_pitch1 - mean_pitch2)

# Calculate the similarity as a percentage
similarity = (1 - euclidean_distance / np.max([mean_pitch1, mean_pitch2])) * 100
print("Similarity between the two audio signals based on pitch features: {:.2f}%".format(similarity))

