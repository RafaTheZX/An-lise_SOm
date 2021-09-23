import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import librosa.display

data_dir = './Analysis_2s/picada/Terra'
audio_files = glob(data_dir + "/*.wav")

for file in range(0, len(audio_files), 1):
    y, sr = librosa.load(audio_files[file])
    n0 = 0
    n1 = 100
    plt.figure(figsize=(14, 5))
    plt.plot(y[n0:n1])
    plt.grid()

    # Zero-crossing counts
    zero_crossings = librosa.zero_crossings(y[n0:n1], pad=False)
    print(sum(zero_crossings))
    print(plt.show())
