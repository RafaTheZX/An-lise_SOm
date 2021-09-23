import numpy as np
import pandas as pd
from matplotlib import pyplot
from glob import glob
import librosa as lib

data_dir = './analise2/pic'
# data_dir = './SomBrutoBatidaWAV/Top_5_mais_limpos'
# data_dir = './Analysis_2s/Batida/Asfalto'
audio_files = glob(data_dir + "/*.wav")

for file in range(0, len(audio_files), 1):
    audio, sfr = lib.load(audio_files[file])
    y_percussive = lib.effects.percussive(audio)
    time = np.arange(0, len(audio)) / sfr
    fig, ax = pyplot.subplots()
    ax.plot(time, audio)
    pyplot.title(f'{audio_files[file]}')
    ax.set(xlabel='Time (s)', ylabel='Sound Amplitude')
    print(pyplot.show())
    print(len(y_percussive))
