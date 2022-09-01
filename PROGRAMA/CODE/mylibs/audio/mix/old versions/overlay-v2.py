import librosa
import IPython as ip
import random
import os
import numpy as np
import soundfile as sf
import numpy as np

class OverlayByFolders():
	"""
	Audios found in path_sound are randomly overlapped with 
	audios found in path_noise. It is possible that not all 
	audios in path_noise are used due to random selection. 
	The selected segment of audios from path_noise to be 
	overlapped with audios from path_sound are also randomly 
	selected.
	"""

	def __init__(self, path_sound, path_noise, path_combined, duration):
	    self.PATH_SOUND = path_sound
	    self.PATH_NOISE = path_noise
	    self.PATH_COMBINED = path_combined

	    self.duration = duration

	    self.SEED = None

	def set_seed(self, seed):
		self.SEED = seed
		random.seed(self.SEED)

	def overlay(self, sound_power=1., noise_power=1.):
		sound_files = [ 
			sound_file for sound_file in os.listdir(self.PATH_SOUND) 
		]
		noise_files = [ 
			noise_file for noise_file in os.listdir(self.PATH_NOISE) 
		]

		for sound_file in sound_files:
			sound_file_path = os.path.join(self.PATH_SOUND, sound_file)

			sound, sample_rate_sound = librosa.load(
				sound_file_path, mono=True, duration=None)

			sound_duration = int(librosa.get_duration(
				filename=sound_file_path))

			sound_offset_segments = range(
				10*sample_rate_sound, 
				sound_duration*sample_rate_sound, 
				sample_rate_sound*3
			)

			for seg_num, offset_sound in enumerate(sound_offset_segments):
				noise_file = random.choice(noise_files)

				noise_file_path = os.path.join(self.PATH_NOISE, noise_file)

				noise_duration = int(librosa.get_duration(
					filename=noise_file_path))

				start_time_noise = random.randint(60, noise_duration - 60)

				noise, sample_rate_noise = librosa.load( noise_file_path, 
					mono=True, offset=start_time_noise, duration=self.duration)

				sound_segment = \
					sound[offset_sound:offset_sound+self.duration*sample_rate_sound]

				# if (offset_sound+duration*sample_rate_sound) > \
				# (sound_duration*sample_rate_sound):
				if (offset_sound + self.duration*sample_rate_sound) > (len(sound)):
					break

				assert len(noise) == len(sound_segment), \
					f'Error lenghts are diferent. \
					{len(noise)} != {len(sound_segment)}.'

				combined = (sound_power*sound_segment + noise_power*noise)/\
					       (sound_power+noise_power)
				combined_name = f'[{noise_file[:-4]}]-' + \
					f'{sound_file[:-4]}.segment{seg_num}.wav'

				sf.write(
					os.path.join(self.PATH_COMBINED, combined_name), 
					combined, sample_rate_sound, 'PCM_16'
				)