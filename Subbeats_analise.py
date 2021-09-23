from glob import glob
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import scipy.stats

y, sr = librosa.load('./analise2/pic/pic_46_asfalto.wav')
tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=512)
cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=512))
print((cqt))
subseg = librosa.segment.subsegment(cqt, beats, n_segments=2)  # pode colocar o plp no lugar do cqt
subseg_t = librosa.frames_to_time(subseg, sr=sr, hop_length=512)

librosa.display.specshow(librosa.amplitude_to_db(cqt,
                                                 ref=np.max),
                         y_axis='cqt_hz', x_axis='time')
lims = plt.gca().get_ylim()
plt.vlines(beat_times, lims[0], lims[1], color='lime', alpha=0.9,
           linewidth=2, label='Beats')
plt.vlines(subseg_t, lims[0], lims[1], color='linen', linestyle='--',
           linewidth=1.5, alpha=0.5, label='Sub-beats')
plt.legend(frameon=True, shadow=True)
plt.title('CQT + Beat and sub-beat markers')
plt.tight_layout()
print(plt.show())
