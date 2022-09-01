import os
import librosa
import librosa.display
import pandas as pd
import numpy as np
from tqdm import tqdm

from configurations.paths import *


class AudioFeatures():
    def __init__(self):
        pass

    # self.PATH_COMBINED_AUDIOS = path_combined_audios
    # self.PATH_CLEANED_AUDIOS = path_cleaned_audios

    @staticmethod
    def load_audio_names(path_raw_audios):
        audio_names, metadata_file = [], None
        for file_name in os.listdir(path_raw_audios):
            if file_name.endswith('wav'):
                audio_names.append(file_name)
            elif file_name.endswith('csv'):
                metadata_file = file_name

        return audio_names, metadata_file

    @staticmethod
    def extract_stft(audio_path, offset=0, duration=None, n_fft=2048, hop_length=None, window="hann", sr=22050):
        try:
            wave, sr = librosa.load(audio_path, offset=offset, duration=duration, sr=22050)
            stft = librosa.stft(wave, n_fft=n_fft, hop_length=hop_length, window=window)
        except FileNotFoundError as e:
            audio_name = audio_path.split('\\')[-1]
            print(e)
            csvfile = os.path.join(PATH_COMBINED, 'metadata.csv')
            df = pd.read_csv(csvfile)
            df = df[df.file_name != audio_name]
            df.to_csv(csvfile, index=False)
            print(f"{audio_name} has been deleted.")

            return None

        # return S_dB
        return stft

    def extract_and_save_stfts_noise(self, ID_list, preprocessor,
                                     path_nosie, path_combined,
                                     type_data='train',
                                     n_fft=2048, hop_length=1024, window="hann",
                                     sr=22050):
        if type_data not in ['train', 'validation']:
            print("Error type_data are not on of the possible values: ", ['train', 'validation'])
            return None

        metadata_file = [f for f in os.listdir(path_combined) if f.endswith('.csv')]
        with open(os.path.join(path_combined, metadata_file[0]), 'r') as csvfile:
            df = pd.read_csv(csvfile)

            for ID in tqdm(ID_list):
                y = df[df['file_name'] == ID + '.wav']
                audio_path = os.path.join(path_nosie, y['noise_name'].iloc[0])
                stft = self.extract_stft(audio_path,
                                         offset=y['noise_offset'].iloc[0],
                                         duration=y['duration'].iloc[0],
                                         n_fft=n_fft, hop_length=hop_length, window=window,
                                         sr=sr)
                stft = np.abs(stft)
                stft = librosa.amplitude_to_db(stft)
                stft = preprocessor.normalise(stft)

                stft = np.expand_dims(stft, axis=2)

                audio_combined_name = y['file_name'].iloc[0]
                save_name_file = ''.join(audio_combined_name.split('.')[:-3])

                # Save spectograms
                with open(os.path.join(PATH_STFTS, type_data, save_name_file, audio_combined_name[:-4],
                                       'noise.npy'), 'wb') as file:
                    np.save(file, stft)

    def extract_and_save_stfts(self, path_combined_audios, path_cleaned_audios, path_stfts,
                               preprocessor,
                               n_fft=2048, hop_length=None, window="hann", sr=22050):
        audio_names, metadata_file = self.load_audio_names(path_combined_audios)
        audio_names, _ = self.load_audio_names(path_cleaned_audios)

        with open(os.path.join(path_combined_audios, metadata_file), 'r') as csvfile:
            df = pd.read_csv(csvfile)
            for _, row in tqdm(df.iterrows()):
                audio_combined_name, audio_cleaned_name, offset, duration = \
                    row['file_name'], row['sound_name'], row['sound_offset'], row['duration']

                save_name_file = ''.join(audio_combined_name.split('.')[:-3])
                # save_name_file = audio_combined_name[:-3]

                audio_combined_path = os.path.join(path_combined_audios, save_name_file, audio_combined_name)
                audio_cleaned_path = os.path.join(path_cleaned_audios, audio_cleaned_name)

                # Exract the stft's
                stft_combined = self.extract_stft(audio_combined_path,
                                                  n_fft=n_fft, hop_length=hop_length, window=window)
                stft_cleaned = self.extract_stft(audio_cleaned_path, offset=offset, duration=duration,
                                                 n_fft=n_fft, hop_length=hop_length, window=window)

                stft_combined, stft_cleaned = np.abs(stft_combined), np.abs(stft_cleaned)
                stft_combined, stft_cleaned = librosa.amplitude_to_db(stft_combined), \
                                              librosa.amplitude_to_db(stft_cleaned)

                stft_combined = preprocessor.normalise(stft_combined)
                stft_cleaned = preprocessor.normalise(stft_cleaned)

                stft_combined = np.expand_dims(stft_combined, axis=2)
                stft_cleaned = np.expand_dims(stft_cleaned, axis=2)

                # Save spectograms
                # verify and create the root directory of spectograms
                if not os.path.exists(os.path.join(path_stfts, save_name_file, audio_combined_name[:-4])):
                    os.makedirs(os.path.join(path_stfts, save_name_file, audio_combined_name[:-4]))

                # save melspectograms files
                with open(os.path.join(path_stfts, 'train', save_name_file, audio_combined_name[:-4], 'combined.npy'),
                          'wb') as file:
                    np.save(file, stft_combined)

                with open(os.path.join(path_stfts, 'train', save_name_file, audio_combined_name[:-4], 'cleaned.npy'),
                          'wb') as file:
                    np.save(file, stft_cleaned)

    def extract_combined_audios_stfts(self,
                                      path_combined_audios, path_cleaned_audios, path_stfts,
                                      n_fft=2048, hop_length=None, window="hann", sr=22050,
                                      preprocessor=None):
        audio_names, metadata_file = self.load_audio_names(path_combined_audios)
        audio_names, _ = self.load_audio_names(path_cleaned_audios)

        with open(os.path.join(path_combined_audios, metadata_file), 'r') as csvfile:
            df = pd.read_csv(csvfile)

            for _, row in tqdm(df.iterrows()):
                audio_combined_name, audio_cleaned_name, offset, duration = \
                    row['file_name'], row['sound_name'], row['sound_offset'], row['duration']

                save_name_file = ''.join(audio_combined_name.split('.')[:-3])
                # save_name_file = audio_combined_name[:-3]

                audio_combined_path = os.path.join(path_combined_audios, save_name_file, audio_combined_name)
                audio_cleaned_path = os.path.join(path_cleaned_audios, audio_cleaned_name)

                stft_combined = self.extract_stft(audio_combined_path,
                                                  n_fft=n_fft, hop_length=hop_length, window=window)
                stft_cleaned = self.extract_stft(audio_cleaned_path, offset=offset, duration=duration,
                                                 n_fft=n_fft, hop_length=hop_length, window=window)

                if preprocessor is not None:
                    stft_combined, stft_cleaned = np.abs(stft_combined), np.abs(stft_cleaned)
                    stft_combined, stft_cleaned = librosa.amplitude_to_db(stft_combined), \
                                                  librosa.amplitude_to_db(stft_cleaned)

                    stft_combined = preprocessor.normalise(stft_combined)
                    stft_cleaned = preprocessor.normalise(stft_cleaned)

                    stft_combined = np.expand_dims(stft_combined, axis=2)
                    stft_cleaned = np.expand_dims(stft_cleaned, axis=2)

                if stft_combined is None or stft_cleaned is None:
                    continue

                if not os.path.exists(os.path.join(path_stfts, save_name_file, audio_combined_name[:-4])):
                    os.makedirs(os.path.join(path_stfts, save_name_file, audio_combined_name[:-4]))

                with open(os.path.join(path_stfts, save_name_file, audio_combined_name[:-4], 'combined.npy'),
                          'wb') as stft_file:
                    np.save(stft_file, stft_combined)

                with open(os.path.join(path_stfts, save_name_file, audio_combined_name[:-4], 'cleaned.npy'),
                          'wb') as stft_file:
                    np.save(stft_file, stft_cleaned)

            # print(stft_combined.shape)

    def extract_combined_audios_stfts_from_stft(self,
                                                path_stfts, path_melspectrograms,
                                                ID_list, preprocessor,
                                                n_fft=2048, hop_length=1024,
                                                window="hann", sr=22050):

        for id in tqdm(ID_list):
            # Load the stft's
            stft_combined = np.load(os.path.join(path_stfts, id, 'combined.npy'))
            stft_cleaned = np.load(os.path.join(path_stfts, id, 'cleaned.npy'))

            # Extract melspectograms
            S_mel_combined, S_mel_cleaned = self.extract_melspectrogram(stft_combined, stft_cleaned, preprocessor,
                                                                        sr=sr, window=window, hop_length=hop_length,
                                                                        n_fft=n_fft)

            # Save melspectograms
            # verify and create the root directory of melspectograms
            if not os.path.exists(os.path.join(path_melspectrograms, id)):
                os.makedirs(os.path.join(path_melspectrograms, id))

            # save melspectograms files
            with open(os.path.join(path_melspectrograms, id, 'combined.npy'), 'wb') as S_mel_combined_file:
                np.save(S_mel_combined_file, S_mel_combined)

            with open(os.path.join(path_melspectrograms, id, 'cleaned.npy'), 'wb') as S_mel_cleaned_file:
                np.save(S_mel_cleaned_file, S_mel_cleaned)

    @staticmethod
    def extract_melspectrogram(stft_x, stft_y, preprocessor,
                               sr=22050, window='hann', hop_length=1024, n_fft=1024 * 2):
        # Extract magnitudes
        Sx, Sy = np.abs(stft_x), np.abs(stft_y)
        Sx, Sy = np.squeeze(Sx), np.squeeze(Sy)

        # Extract melspectrogram
        Sx_mel = librosa.feature.melspectrogram(
            S=Sx, sr=sr, n_fft=n_fft, hop_length=hop_length, window=window)
        Sy_mel = librosa.feature.melspectrogram(
            S=Sy, sr=sr, n_fft=n_fft, hop_length=hop_length, window=window)

        # Amplitude to dB
        # Sx_mel = librosa.amplitude_to_db(Sx_mel)
        # Sy_mel = librosa.amplitude_to_db(Sy_mel)

        # Normalization
        Sx_mel = preprocessor.normalise(Sx_mel)
        Sy_mel = preprocessor.normalise(Sy_mel)

        # Shape formatting
        Sx_mel = np.expand_dims(Sx_mel, axis=2)
        Sy_mel = np.expand_dims(Sy_mel, axis=2)

        return Sx_mel, Sy_mel  # , S_combined, S_cleaned

    @staticmethod
    def create_validation_dataset(path, percentage=0.2):
        import shutil
        import random

        # num_validation_samples = int(len(path_ids)*percentage)

        path_aux = os.path.join(path, 'train')
        path_ids = []
        for folder in os.listdir(path_aux):
            for file_name in os.listdir(os.path.join(path_aux, folder)):
                path_ids.append(os.path.join(folder, file_name))

        random.shuffle(path_ids)

        num_validation_samples = int(len(path_ids) * percentage)
        for path_id in tqdm(path_ids[:num_validation_samples]):
            path_train = os.path.join(path, 'train', path_id)
            path_validaton = os.path.join(path, 'validation', path_id)

            os.makedirs(path_validaton, exist_ok=True)

            path_src_combined = os.path.join(path_train, 'combined.npy')
            path_dst_combined = os.path.join(path_validaton, 'combined.npy')
            shutil.move(path_src_combined, path_dst_combined)

            path_src_cleaned = os.path.join(path_train, 'cleaned.npy')
            path_dst_cleaned = os.path.join(path_validaton, 'cleaned.npy')
            shutil.move(path_src_cleaned, path_dst_cleaned)

            os.rmdir(path_train)


if __name__ == '__main__':
    from configurations.paths import *
    import os
    from mylibs.audio.features.preprocess import *
    import matplotlib.pyplot as plt

    print("STFT's stage has started.")

    path_manager = PathsManager(3)
    # PATH_COMBINED = path_manager.path_combined
    # PATH_SONGS = path_manager.path_songs
    PATH_STFTS = path_manager.path_stfts
    # PATH_NOISE = path_manager.path_env_audios
    # PATH_MELSPECTROGRAMS = path_manager.path_melspectrograms

    PATH_STFTS_TEST = path_manager.path_stfts_test
    PATH_COMBINED_TEST = path_manager.path_combined_test
    PATH_NOISE_TEST = path_manager.path_env_audios
    PATH_SONGS_TEST = path_manager.path_songs_test
    PATH_MELESPECTROGRAM_TEST = path_manager.path_melspectrograms_test

    audio_features = AudioFeatures()  # initialize object

    # Parameters
    frame_length = 1024 * 2
    overlap_percentage = 0.5

    # Set preprocessor
    preprocessor = MinMaxNormaliser(0, 1)

    # Create ID_list
    ID_list_train = []
    for folder in os.listdir(os.path.join(PATH_STFTS, 'train'))[1:]:
        for file_name in os.listdir(os.path.join(PATH_STFTS, 'train', folder)):
            ID_list_train.append(os.path.join(file_name))

    ID_list_validation = []
    for folder in os.listdir(os.path.join(PATH_STFTS, 'validation'))[1:]:
        for file_name in os.listdir(os.path.join(PATH_STFTS, 'validation', folder)):
            ID_list_validation.append(os.path.join(file_name))

    ID_list_test = []
    for folder in os.listdir(PATH_STFTS_TEST)[1:]:
        for file_name in os.listdir(os.path.join(PATH_STFTS_TEST, folder)):
            ID_list_validation.append(os.path.join(file_name))

    # audio_features.extract_combined_audios_stfts(PATH_COMBINED_TEST, PATH_SONGS_TEST, PATH_STFTS_TEST,
    #                                              n_fft=2048, hop_length=1024, window="hann", sr=22050,
    #                                              preprocessor=preprocessor)

    # Extracting and saving melspectograms
    # audio_features.extract_combined_audios_stfts_from_stft(PATH_STFTS, os.path.join(PATH_MELSPECTROGRAMS, 'train'),
    #                                                        ID_list_train, preprocessor,
    #                                                        n_fft=frame_length,
    #                                                        hop_length=int(frame_length * overlap_percentage),
    #                                                        window="hann", sr=22050)

    # audio_features.extract_combined_audios_stfts(PATH_COMBINED, PATH_SONGS, PATH_STFTS,
    #                                              n_fft=frame_length,
    #                                              hop_length=int(frame_length * overlap_percentage))

    # Create validation dataset
    # audio_features.create_validation_dataset(PATH_MELSPECTROGRAMS)

    # audio_features.extract_and_save_stfts(PATH_COMBINED, PATH_SONGS, PATH_STFTS,
    #                                       preprocessor=preprocessor,
    #                                       n_fft=frame_length,
    #                                       hop_length=int(frame_length * overlap_percentage))
    # audio_features.create_validation_dataset(PATH_STFTS)

    #
    # audio_features.extract_and_save_stfts_noise(ID_list_train, preprocessor,
    #                                             PATH_NOISE, PATH_COMBINED,
    #                                             type_data='train',
    #                                             n_fft=2048, hop_length=1024, window="hann",
    #                                             sr=22050)
    #
    # audio_features.extract_and_save_stfts_noise(ID_list_validation, preprocessor,
    #                                             PATH_NOISE, PATH_COMBINED,
    #                                             type_data='validation',
    #                                             n_fft=2048, hop_length=1024, window="hann",
    #                                             sr=22050)

    # MEL ESPECTROGRAM
    ID_list_aux1 = pd.read_csv(os.path.join(PATH_COMBINED_TEST, 'metadata.csv'))['file_name'].map(
        lambda x: ''.join(x.split('.')[:-3]))
    ID_list_aux2 = pd.read_csv(os.path.join(PATH_COMBINED_TEST, 'metadata.csv'))['file_name'].map(lambda x: x[:-4])
    ID_list_mel_test = list(map(lambda f1, f2: os.path.join(f1, f2), ID_list_aux1, ID_list_aux2))


    audio_features.extract_combined_audios_stfts_from_stft(PATH_STFTS_TEST, PATH_MELESPECTROGRAM_TEST,
                                                           ID_list_mel_test, preprocessor=preprocessor,
                                                           n_fft=2048, hop_length=1024, sr=22050)


    # print(ID_list_train)

    # audio_features.extract_and_save_stfts_noise(ID_list_test, preprocessor,
    #                                             PATH_NOISE_TEST, PATH_COMBINED_TEST,
    #                                             type_data='validation',
    #                                             n_fft=2048, hop_length=1024, window="hann",
    #                                             sr=22050)

    def plot_spectograms(IDs, path=PATH_STFTS, n_fft=2048, hop_length=1024):
        import matplotlib.pyplot as plt
        import librosa

        for i in range(5):
            x = np.load(os.path.join(path, IDs[i], 'combined.npy'))
            y = np.load(os.path.join(path, IDs[i], 'cleaned.npy'))

            print(x.shape)

            x, y = np.abs(x), np.abs(y)
            x, y = librosa.amplitude_to_db(x), librosa.amplitude_to_db(y)

            plt.figure(i)
            fig, ax = plt.subplots(nrows=2, ncols=1)

            ax[0].set_title('Combined')
            librosa.display.specshow(x, x_axis='time', y_axis='linear', ax=ax[0],
                                     n_fft=n_fft, hop_length=hop_length)
            ax[1].set_title('Cleaned')
            librosa.display.specshow(y, x_axis='time', y_axis='linear', ax=ax[1],
                                     n_fft=n_fft, hop_length=hop_length)

    # np.random.shuffle(ID_list_train)
    # plot_spectograms(ID_list_train[:5])
    # plt.show()
