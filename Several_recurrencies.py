import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import librosa.display

data_dir = './Analysis_2s/picada/Terra'
audio_files = glob(data_dir + "/*.wav")

for file in range(0, len(audio_files), 1):
    y, sr = librosa.load(audio_files[file])
    C = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma = librosa.feature.chroma_cqt(C=C, sr=sr)
    onset_envelope = librosa.onset.onset_strength(y, sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_envelope)
    c_sync = librosa.util.sync(chroma, beats, aggregate=np.median)
    chroma_stack = librosa.feature.stack_memory(c_sync, n_steps=4, mode='edge')
    R = librosa.segment.recurrence_matrix(chroma_stack, sym=True)

    plt.subplot()
    librosa.display.specshow(R, x_axis='time', y_axis='time', hop_length=1024)
    plt.title(f'{audio_files[file]}')
    plt.tight_layout()

    print(plt.show())