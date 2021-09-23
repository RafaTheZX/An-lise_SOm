import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import librosa.display

data_dir = './Croc_croc/Tseca'
audio_files = glob(data_dir + "/*.wav")

for file in range(0, len(audio_files), 1):
    y, sfr = librosa.load(audio_files[file], duration=1.0)
    plt.figure(figsize=(12, 8))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.subplot(4, 2, 1)
    librosa.display.specshow(D, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'{audio_files[file]}')
    print(plt.show())
