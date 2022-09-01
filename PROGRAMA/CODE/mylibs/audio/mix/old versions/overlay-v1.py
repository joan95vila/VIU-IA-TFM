import librosa
import IPython as ip
import random
import os
import numpy as np
import soundfile as sf
import numpy as np

class OverlayByFolders():
	"""
	Audios found in path_1 are randomly overlapped with 
	audios found in path_2. It is possible that not all 
	audios in path_2 are used due to random selection. 
	The selected segment of audios from path_2 to be 
	overlapped with audios from path_1 are also randomly 
	selected.
	"""

	def __init__(self, path_1, path_2, path_combined, duration):
	    self.PATH_AUDIOS_1 = path_1
	    self.PATH_AUDIOS_2 = path_2
	    self.PATH_COMBINED = path_combined

	    self.duration = duration

	    self.SEED = None

	def set_seed(self, seed):
		self.SEED = seed
		random.seed(self.SEED)

	def overlay(self, audio_1_power=1., audio_2_power=1.):
		audio_1_files = [ 
			audio_file for audio_file in os.listdir(self.PATH_AUDIOS_1) 
		]
		audio_2_files = [ 
			audio_file for audio_file in os.listdir(self.PATH_AUDIOS_2) 
		]

		for audio_1_file in audio_1_files:
			audio_1_file_path = os.path.join(self.PATH_AUDIOS_1, audio_1_file)

			audio_1, sample_rate_audio_1 = librosa.load(
				audio_1_file_path, mono=True, duration=None)

			audio_1_duration = int(librosa.get_duration(
				filename=audio_1_file_path))

			audio_1_offset_segments = range(
				10*sample_rate_audio_1, 
				audio_1_duration*sample_rate_audio_1, 
				sample_rate_audio_1*3
			)

			for seg_num, offset_audio_1 in enumerate(audio_1_offset_segments):
				audio_2_file = random.choice(audio_2_files)

				audio_2_file_path = os.path.join(self.PATH_AUDIOS_2, audio_2_file)

				audio_2_duration = int(librosa.get_duration(
					filename=audio_2_file_path))

				start_time_audio_2 = random.randint(60, audio_2_duration - 60)

				audio_2, sample_rate_audio_2 = librosa.load( audio_2_file_path, 
					mono=True, offset=start_time_audio_2, duration=self.duration)

				audio_1_segment = \
					audio_1[offset_audio_1:offset_audio_1+self.duration*sample_rate_audio_1]

				# if (offset_audio_1+duration*sample_rate_audio_1) > \
				# (audio_1_duration*sample_rate_audio_1):
				if (offset_audio_1 + self.duration*sample_rate_audio_1) > (len(audio_1)):
					break

				assert len(audio_2) == len(audio_1_segment), \
					f'Error lenghts are diferent. \
					{len(audio_2)} != {len(audio_1_segment)}.'

				combined = (audio_1_power*audio_1_segment + audio_2_power*audio_2)/\
					       (audio_1_power+audio_2_power)
				combined_name = f'[{audio_2_file[:-4]}]-' + \
					f'{audio_1_file[:-4]}.segment{seg_num}.wav'

				sf.write(
					os.path.join(self.PATH_COMBINED, combined_name), 
					combined, sample_rate_audio_1, 'PCM_16'
				)