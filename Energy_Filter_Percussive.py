from glob import glob
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

data_dir = './Croc_croc/Tseca'
audio_files = glob(data_dir + "/*.wav")

for file in range(0, len(audio_files), 1):
    y, sr = librosa.load(audio_files[file], duration=1.0)
    print(len(y))
    y_percussive = librosa.effects.percussive(y)
    print((len(y_percussive)))
    S = librosa.feature.melspectrogram(y_percussive, sr=sr, power=1)
    print(len(S))
    log_S = librosa.amplitude_to_db(S, ref=np.max)
    pcen_S = librosa.pcen(S * (2**31))
    print((len(pcen_S)))
    plt.figure()
    plt.subplot(2, 1, 1)
    librosa.display.specshow(log_S, x_axis='time', y_axis='mel')
    plt.title('log amplitude (dB)')
    plt.colorbar()
    plt.subplot(2, 1, 2)
    librosa.display.specshow(pcen_S, x_axis='time', y_axis='mel')
    plt.title('Filtro de Energia Percussivo')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
