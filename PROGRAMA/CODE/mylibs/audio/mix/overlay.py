import csv
import os
import random
import librosa
import numpy as np
import soundfile as sf

from tqdm import tqdm


class Overlay:
    """
    Audios found in path_sound are randomly overlapped with
    audios found in path_noise. It is possible that not all
    audios in path_noise are used due to random selection.
    The selected fragments of audios from path_noise to be
    overlapped with audios from path_sound are also randomly
    selected.
    """

    def __init__(self, path_sound, path_noise, path_combined, duration,
                 sound_start_time=10, noise_start_time=60, noise_end_time=60):
        self.PATH_SOUND = path_sound
        self.PATH_NOISE = path_noise
        self.PATH_COMBINED = path_combined

        self.duration = duration
        self.sound_start_time = sound_start_time
        self.noise_start_time = noise_start_time
        self.noise_end_time = noise_end_time

        self.SEED = None

    def set_seed(self, seed):
        self.SEED = seed
        random.seed(self.SEED)

    def overlay(self, sound_file, noise_file,
                sound_power=1., noise_power=1.,
                sound_offset=0, noise_offset=0,
                save_name=None):

        # setting paths
        sound_file_path = os.path.join(self.PATH_SOUND, sound_file)
        noise_file_path = os.path.join(self.PATH_NOISE, noise_file)

        # setting sound variables
        sound, sample_rate_sound = librosa.load(sound_file_path,
                                                offset=sound_offset, duration=self.duration)

        sound_duration = int(librosa.get_duration(
            filename=sound_file_path))

        # setting noise variables
        noise, sample_rate_noise = librosa.load(noise_file_path,
                                                offset=noise_offset, duration=self.duration)

        if not save_name:
            combined_name = f'[{noise_file[:-4]}]-' + \
                            f'{sound_file[:-4]}.wav'

        if (sound_offset + self.duration) > sound_duration:
            f'[WARNING]: A sound (started in {sound_offset + self.duration, sound_duration}) was omitted.'

        else:
            assert len(noise) == len(sound), \
                f'Error lengths are different. {len(noise)} != {len(sound)}.'

            combined = (sound_power * sound + noise_power * noise) / \
                       (sound_power + noise_power)

            save_name_file = ''.join(save_name.split('.')[:-3])

            if not os.path.exists(os.path.join(self.PATH_COMBINED, save_name_file)):
                os.makedirs(os.path.join(self.PATH_COMBINED, save_name_file))

            sf.write(
                os.path.join(self.PATH_COMBINED, save_name_file, save_name),
                combined, sample_rate_sound, 'PCM_16'
            )

    def overlay_by_folders_sequential(self, sound_power=1., noise_power=1., max_songs=None):
        sound_files = [
            sound_file for sound_file in os.listdir(self.PATH_SOUND)[:max_songs]
        ]
        noise_files = [
            noise_file for noise_file in os.listdir(self.PATH_NOISE)
        ]

        with open(os.path.join(self.PATH_COMBINED, 'metadata.csv'), 'a') as csvfile:
            fieldnames = ['file_name', 'duration', 'sound_name',
                          'noise_name', 'sound_offset', 'noise_offset',
                          'sound_power', 'noise_power']

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # iterate over each sound file
            for sound_file in tqdm(sound_files):
                sound_file_path = os.path.join(self.PATH_SOUND, sound_file)

                sound_duration = int(librosa.get_duration(
                    filename=sound_file_path))

                sound_offset = np.arange(
                    self.sound_start_time, sound_duration, self.duration
                )

                # iterate over each sound fragment
                for fragment_id, sound_offset in enumerate(sound_offset):
                    noise_file = random.choice(noise_files)  # select a random noise

                    noise_file_path = os.path.join(self.PATH_NOISE, noise_file)

                    noise_duration = int(librosa.get_duration(
                        filename=noise_file_path))

                    noise_offset = random.randint(
                        self.noise_start_time,
                        noise_duration - self.noise_end_time
                    )

                    combined_name = f'[{noise_file[:-4]}]-' + \
                                    f'{sound_file[:-4]}.{fragment_id}.wav'

                    if not (sound_offset + self.duration) > sound_duration:
                        writer.writerow({
                            'file_name': combined_name,
                            'duration': self.duration,
                            'sound_name': sound_file,
                            'noise_name': noise_file,
                            'sound_offset': sound_offset,
                            'noise_offset': noise_offset,
                            'sound_power': sound_power,
                            'noise_power': noise_power
                        })

                    self.overlay(
                        sound_file, noise_file,
                        sound_offset=sound_offset, noise_offset=noise_offset,
                        sound_power=sound_power, noise_power=noise_power,
                        save_name=combined_name,
                    )


if __name__ == '__main__':
    from configurations.paths import *
    import os

    path_manager = PathsManager(3)
    # PATH_COMBINED = path_manager.path_combined
    # PATH_ENV_AUDIOS = path_manager.path_env_audios
    # PATH_SONGS = path_manager.path_songs
    # PATH_STFTS = path_manager.path_stfts

    PATH_COMBINED_TEST = path_manager.path_combined_test
    PATH_ENV_AUDIOS_TEST = path_manager.path_env_audios
    PATH_SONGS_TEST = path_manager.path_songs_test
    PATH_STFTS_TEST = path_manager.path_stfts_test

    print("Overlay stage has started.")

    segment_duration = 0.74  # seconds

    sound_start_time = 10
    noise_start_time = 60
    noise_end_time = 60

    sound_power = 0.5
    noise_power = 1
    max_songs = None


    # overlay = Overlay(PATH_SONGS, PATH_ENV_AUDIOS, PATH_COMBINED, segment_duration, sound_start_time=sound_start_time,
    #                   noise_start_time=noise_start_time, noise_end_time=noise_end_time)  # initialize object
    overlay = Overlay(PATH_SONGS_TEST, PATH_ENV_AUDIOS_TEST, PATH_COMBINED_TEST, segment_duration, sound_start_time=sound_start_time,
                      noise_start_time=noise_start_time, noise_end_time=noise_end_time)  # initialize object
    overlay.overlay_by_folders_sequential(sound_power=sound_power, noise_power=noise_power, max_songs=max_songs)  # overlay