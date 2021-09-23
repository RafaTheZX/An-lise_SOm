import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import librosa.display

data_dir = './Croc_croc/Tseca'
audio_files = glob(data_dir + "/*.wav")

for file in range(0, len(audio_files), 1):
    y, sr = librosa.load(audio_files[file], duration=1.0)
    y_percussive = librosa.effects.percussive(y)
    hop_length = 512
    crm = librosa.feature.chroma_stft(y_percussive, sr=sr, hop_length=hop_length)
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(crm, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')
    plt.colorbar()
    plt.title(f'{audio_files[file]}')
    print(plt.show())
