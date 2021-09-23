import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import librosa.display

# Para analisar o ritmo da música, temos que encontrar os beats

data_dir = './Croc_croc/Tseca'
audio_files = glob(data_dir + "/*.wav")

for file in range(0, len(audio_files), 1):
    y, sr = librosa.load(audio_files[file], duration=1.0)
    y_percussive = librosa.effects.percussive(y)  # os surdos captam apenas o aspecto percussivo do som
    onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr, aggregate=np.median)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    librosa.frames_to_time(beats[:4], sr=sr)
    hop_length = 512
    plt.figure(figsize=(8, 4))
    times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
    print(onset_env[beats])
    print(max(onset_env[beats]))
    n = np.divide(onset_env[beats], max(onset_env[beats]))
    print(n) # vetor intensidade da relação
    print(times[beats]) # aqui fica o vetor tempo para programar a vibração do feel de music
    plt.plot(times, librosa.util.normalize(onset_env), label='Onset strength')
    plt.vlines(times[beats], 0, 1, alpha=0.5, color='r', linestyle='--', label='Beats')
    plt.legend(frameon=True, framealpha=0.75)
    print(plt.show())

# Cada linha vermelha é quando o celular vai vibrar