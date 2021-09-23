import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import librosa.display

data_dir = './Marcha_Picada_WAV/Mais_Sujos'
audio_files = glob(data_dir + "/*.wav")

for file in range(0, len(audio_files), 1):
    y, sr = librosa.load(audio_files[file])
    plt.figure(figsize=(12, 8))
    C = librosa.feature.chroma_cqt(y=y, sr=sr)
    plt.subplot()
    librosa.display.specshow(C, x_axis='time', y_axis='chroma')
    plt.colorbar()
    plt.title(f'{audio_files[file]}')
    plt.tight_layout()
    print(plt.show())
