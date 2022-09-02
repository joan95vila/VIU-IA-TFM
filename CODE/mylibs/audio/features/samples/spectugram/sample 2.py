from configurations.paths import *
from myaudio.mix.overlay import *
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


# overlay = OverlayByFolders(PATH_SONGS, PATH_ENV_AUDIOS, PATH_COMBINED, duration=3)
# overlay.overlay(audio_1_power=0.5, audio_2_power=1.0)

# PATHS
song_path = os.path.join(PATH_SONGS, os.listdir(PATH_SONGS)[0])
noise_path = os.path.join(PATH_ENV_AUDIOS, os.listdir(PATH_ENV_AUDIOS)[0])

# combined_path = os.path.join(PATH_COMBINED, os.listdir(PATH_COMBINED)[0])

# CONFIGURATIONS
duration = [1, 3, 6]
offset = 60

# (wave, noise, combined) = ([1,2,3], 4, 5)

# LOAD AUDIOS
audio = np.empty((len(duration), 3), dtype=object)
stft = np.empty_like(audio)

# audio[0, 0] = [9,9]

# print(audio)
# exit()

# figure 1
# wavelength song, noise, combined * nrows of durations

# figure 2
# spectogram song, noise, combined * nrows of duration

for i, d in enumerate(duration):
	wave, sr = librosa.load(song_path, offset=offset, mono=True, duration=d)
	noise, sr = librosa.load(noise_path, offset=offset, mono=True, duration=d)

	combined = wave + noise

	audio[i][:] = wave, noise, combined

	nested_map = lambda x: librosa.power_to_db(np.abs(librosa.stft(x)))
	stft[i][:] = list(map(nested_map, (wave, noise, combined)))

# print(stft)

fig, ax = plt.subplots(nrows=len(duration), ncols=3, sharex=False, sharey=True)
for i, d in enumerate(duration):
	# print(stft[i, 0])
	# print(stft[i, 0].shape)
	ax[i, 0].set_title('Wave')
	ax[i, 1].set_title('Noise')
	ax[i, 2].set_title('Combined')
	librosa.display.specshow(stft[i, 0], x_axis='time', y_axis='log', ax=ax[i, 0])
	librosa.display.specshow(stft[i, 1], x_axis='time', y_axis='log', ax=ax[i, 1])
	librosa.display.specshow(stft[i, 2], x_axis='time', y_axis='log', ax=ax[i, 2])

# plt.show()

plt.figure(2)
fig, ax = plt.subplots(nrows=len(duration), ncols=3, sharex=False, sharey=True)
for i, d in enumerate(duration):
	ax[i, 0].set_title('Wave')
	ax[i, 1].set_title('Noise')
	ax[i, 2].set_title('Combined')

	print(audio[i, 0])

	ax[i, 0].plot(audio[i, 0])
	ax[i, 1].plot(audio[i, 1])
	ax[i, 2].plot(audio[i, 2])

plt.show()