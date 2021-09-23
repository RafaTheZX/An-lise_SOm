import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import librosa.display
import sklearn

data_dir = './Analysis_2s/picada/Terra'
audio_files = glob(data_dir + "/*.wav")

for file in range(0, len(audio_files), 1):
    y, sr = librosa.load(audio_files[file])
    spectral_centroids = librosa.feature.spectral_centroid(y, sr=sr)[0]
    spectral_centroids.shape

    # Computing the time variable for visualization
    plt.figure(figsize=(12, 4))
    frames = range(len(spectral_centroids))
    t = librosa.frames_to_time(frames)

    def normalize(x, axis=0):
        return sklearn.preprocessing.minmax_scale(x, axis=axis)

    librosa.display.waveplot(y, sr=sr, alpha=0.4)
    plt.plot(t, normalize(spectral_centroids), color='r')
    print(plt.show())
