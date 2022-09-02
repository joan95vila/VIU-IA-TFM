from configurations.paths import *
from myaudio.mix.overlay import *
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


# overlay = OverlayByFolders(PATH_SONGS, PATH_ENV_AUDIOS, PATH_COMBINED, duration=3)
# overlay.overlay(audio_1_power=0.5, audio_2_power=1.0)


audio_path = os.path.join(PATH_SONGS, os.listdir(PATH_SONGS)[0])
combined_path = os.path.join(PATH_COMBINED, os.listdir(PATH_COMBINED)[0])

print(combined_path)
print(audio_path)

# exit()

i = 4

audio, sr = [[]]*i, [[]]*i
stft, stft_magnitud = [[]]*i, [[]]*i

fig, ax = plt.subplots(nrows=4, ncols=1, sharex=False)

for i, d in enumerate([1,3,6]):
	audio[i], sr[i] = librosa.load(audio_path, offset=10, mono=True, duration=d)

	# stft = np.abs(np.fft.fft(audio))
	stft[i] = librosa.stft(audio[i])
	stft_magnitud[i] = np.abs(stft[i])

	librosa.display.specshow(librosa.power_to_db(stft_magnitud[i]), x_axis='time', y_axis='log', ax=ax[i])  # spectrogram in row i, column 2


i = 3
audio[i], sr[i] = librosa.load(combined_path, mono=True, duration=3)

stft[i] = librosa.stft(audio[i])
stft_magnitud[i] = np.abs(stft[i])

librosa.display.specshow(librosa.power_to_db(stft_magnitud[i]), x_axis='time', y_axis='log', ax=ax[i])


plt.show()