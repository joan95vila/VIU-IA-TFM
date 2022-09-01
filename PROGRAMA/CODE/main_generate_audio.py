from mylibs.audio.features.stft import *
from mylibs.audio.mix.overlay import *
from configurations.paths import *
from configurations.params import *
from mylibs.audio.features.preprocess import *
from mylibs.ml.autoencoder import VAE
from mylibs.ml.train import *
from mylibs.utilities.datagen import *
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
import numpy as np
import os
import tensorflow as tf


# PATH set up
path_manager = PathsManager(1)
PATH_COMBINED = path_manager.path_combined
PATH_ENV_AUDIOS = path_manager.path_env_audios
PATH_SONGS = path_manager.path_songs
PATH_STFTS = path_manager.path_stfts
PATH_MELSPECTROGRAMS = path_manager.path_melspectrograms

# Parameters
for key, val in params_dict.items():
    exec(key + '=val')

params = {
    'dim': list(input_shape)[:-1],
    'batch_size': batch_size,
    'n_channels': 1,
    'shuffle': shuffle,
    'preprocessor': preprocessor(min_normalize_value, max_normalize_value),
    'sr': sample_rate,
    'n_fft': frame_length,
    'hop_length': int(frame_length * overlap_percentage),
    'window': window,
    'seed': seed
}


def generate_audio(model, audio_path, noisy_aduio_path,
                   offset_audio=0, offset_noisy=0, duration=0,
                   sr=22050, window='hann', hop_length=1024, n_fft=1024 * 2,
                   play=True, save=True, type_data='spectogram', UNET=False):
    import simpleaudio as sa

    print("Generating audio has started.")

    audio_features = AudioFeatures()
    preprocessor = MinMaxNormaliser(0, 1)

    noisy_audio = librosa.load(os.path.join(PATH_COMBINED, noisy_aduio_path + '.wav'))[0]
    audio = librosa.load(audio_path, sr=sr, offset=offset_audio, duration=duration)[0]
    stft = librosa.stft(noisy_audio, window=window, hop_length=hop_length, n_fft=n_fft)
    phase = np.angle(stft)
    magnitude = np.abs(stft)

    if UNET:
        phase = phase[:-1, ...]
        magnitude = magnitude[:-1, ...]
        # stft = stft[:-1, ...]

    if type_data == 'mel':
        melspectrogram = librosa.feature.melspectrogram(S=magnitude, sr=sr, window=window,
                                                        hop_length=hop_length, n_fft=n_fft)
        melspectrogram = librosa.amplitude_to_db(melspectrogram)
        melspectrogram = preprocessor.normalise(melspectrogram)

        S = melspectrogram

        noisy_image = np.load(os.path.join(PATH_MELSPECTROGRAMS, 'train', noisy_aduio_path, 'combined.npy'))
        noisy_image = np.squeeze(noisy_image)
        clean_image = np.load(os.path.join(PATH_MELSPECTROGRAMS, 'train', noisy_aduio_path, 'cleaned.npy'))
        clean_image = np.squeeze(clean_image)

    else:
        S = librosa.amplitude_to_db(magnitude)
        S = preprocessor.normalise(S)

        noisy_image = np.load(os.path.join(PATH_STFTS, 'train', noisy_aduio_path, 'combined.npy'))
        if UNET:
            noisy_image = noisy_image[:-1, ...]
        noisy_image = np.squeeze(noisy_image)
        clean_image = np.load(os.path.join(PATH_STFTS, 'train', noisy_aduio_path, 'cleaned.npy'))
        if UNET:
            clean_image = clean_image[:-1, ...]
        clean_image = np.squeeze(clean_image)

    S = np.expand_dims(S, axis=[0, 3])

    if UNET:
        reconstructed_image = model.predict(S)
    else:
        reconstructed_image, latent_representation = model.reconstruct(S)

    reconstructed_image = np.squeeze(reconstructed_image)

    def polarize(x):
        # print(np.max(reconstructed_image), np.max(reconstructed_image)*0.7)
        return x**1.23
        # if x > np.max(reconstructed_image)*0.7:
        #     return x
        # else:
        #     return x*0.0

    # apply_all = np.vectorize(polarize)
    # reconstructed_image_polarized = apply_all(reconstructed_image)
    # reconstructed_image_polarized = preprocessor.normalise(reconstructed_image_polarized)

    S = preprocessor.denormalise(reconstructed_image, preprocessor.min, preprocessor.max)
    S = librosa.db_to_amplitude(S)
    # S_polarized = preprocessor.denormalise(reconstructed_image_polarized, preprocessor.min, preprocessor.max)
    # S_polarized = librosa.db_to_amplitude(S_polarized)
    if type_data == 'mel':
        S = librosa.feature.inverse.mel_to_stft(S, n_fft=n_fft)

    stft = S * (np.cos(phase) + np.sin(phase) * 1j)
    # stft_polarized = S_polarized * (np.cos(phase) + np.sin(phase) * 1j)
    if UNET:
        reconstructed_audio = librosa.istft(stft, n_fft=n_fft-1, hop_length=hop_length, window=window,
                                            length=16317)
    else:
        reconstructed_audio = librosa.istft(stft, n_fft=n_fft, hop_length=hop_length, window=window,
                                            length=16317)
    # reconstructed_audio_polarized = librosa.istft(stft_polarized, n_fft=n_fft, hop_length=hop_length, window=window,
    #                                     length=16317)

    # combined = 0.5*cleaned + noise
    subtracted_audio = noisy_audio - reconstructed_audio  # combined - cleaned'
    subtracted_audio = (noisy_audio - subtracted_audio)*2
    # subtracted_audio = reconstructed_audio - subtracted_audio
    # cleaned' - (combined - cleaned') = 2*cleaned' - combined
    # 2*cleaned' - combined = 2*cleaned' - cleaned - noise ~ cleaned - noise
    # print('[original audio] shape: ', audio.shape)
    # print('[reconstructed audio] shape: ', reconstructed_audio.shape)
    # print('[subtracted audio] shape: ', subtracted_audio.shape)

    play = play
    if play:
        print("Playing the noisy audio...")
        play_obj3 = sa.play_buffer(noisy_audio, 1, 4, sr)
        play_obj3.wait_done()
        print("Playing the clean audio...")
        play_obj1 = sa.play_buffer(audio, 1, 4, sr)
        play_obj1.wait_done()
        print("Playing the reconstructed audio...")
        reconstructed_audio = (audio * 100) * (reconstructed_audio * 1)
        play_obj2 = sa.play_buffer(reconstructed_audio, 1, 4, sr)
        play_obj2.wait_done()
        # print("Playing the reconstructed polarized audio...")
        # play_obj = sa.play_buffer(reconstructed_audio_polarized, 1, 4, sr)
        # play_obj.wait_done()
        print("Playing the noisy audio with the reconstructed audio subtracted...")
        play_obj4 = sa.play_buffer(subtracted_audio, 1, 4, sr)
        play_obj4.wait_done()

    save = save
    if save:
        sf.write('model/test/clean.wav', audio, sr)
        sf.write('model/test/combined.wav', noisy_audio, sr)
        sf.write('model/test/reconstructed.wav', reconstructed_audio, sr)
        sf.write('model/test/combined_sub_reconstructed.wav', subtracted_audio, sr)

    def find_dpi(w, h, d):
        """
        w : width in pixels
        h : height in pixels
        d : diagonal in inches
        """
        w_inches = (d ** 2 / (1 + h ** 2 / w ** 2)) ** 0.5
        return round(w / w_inches)

    dpi = find_dpi(3440, 1440, 34)

    fig, axs = plt.subplots(nrows=6, ncols=1, dpi=dpi, figsize=(76 * 0.39 / 2, 31.5 * 0.39 / 2))
    # fig, axs = plt.subplots(nrows=3, ncols=1, dpi=dpi, figsize=(76 * 0.39 / 2, 31.5 * 0.39 / 2))

    if UNET:
        axs[0].set_title('COMBINED')
        img0 = librosa.display.specshow(noisy_image, x_axis='time', y_axis='linear', ax=axs[0],
                                 n_fft=frame_length-1, hop_length=frame_length // 2)

        axs[1].set_title('CLEANED')
        img1 = librosa.display.specshow(clean_image, x_axis='time', y_axis='linear', ax=axs[1],
                                       n_fft=frame_length-1, hop_length=frame_length // 2)

        axs[2].set_title('PREDICTION')
        img2 = librosa.display.specshow(reconstructed_image, x_axis='time', y_axis='linear', ax=axs[2],
                                 n_fft=frame_length-1, hop_length=frame_length // 2)

        axs[3].set_title('RECONSTRUCTED WAVEFORM')
        axs[3].plot(reconstructed_audio)

        axs[4].set_title('WAVEFORM CLEAN')
        axs[4].plot(audio)

        axs[5].set_title('WAVEFORM COMBINED')
        axs[5].plot(noisy_audio)

        # print(audio.shape, reconstructed_audio.shape)

        # axs[3].set_title('PREDICTION POLARIZED')
        # img3 = librosa.display.specshow(reconstructed_image_polarized, x_axis='time', y_axis='linear', ax=axs[3],
        #                                n_fft=frame_length-1, hop_length=frame_length // 2)
    else:
        axs[0].set_title('COMBINED')
        img0 = librosa.display.specshow(noisy_image, x_axis='time', y_axis='linear', ax=axs[0],
                                        n_fft=frame_length, hop_length=frame_length // 2)

        axs[1].set_title('CLEANED')
        img1 = librosa.display.specshow(clean_image, x_axis='time', y_axis='linear', ax=axs[1],
                                        n_fft=frame_length, hop_length=frame_length // 2)

        axs[2].set_title('PREDICTION')
        img2 = librosa.display.specshow(reconstructed_image, x_axis='time', y_axis='linear', ax=axs[2],
                                        n_fft=frame_length, hop_length=frame_length // 2)

        axs[3].set_title('PREDICTION POLARIZED')
        img3 = librosa.display.specshow(reconstructed_image_polarized, x_axis='time', y_axis='linear', ax=axs[3],
                                        n_fft=frame_length, hop_length=frame_length // 2)

    # fig.colorbar(img, ax=axs[2], orientation='horizontal')
    fig.colorbar(img0, ax=axs[0])
    fig.colorbar(img1, ax=axs[1])
    fig.colorbar(img2, ax=axs[2])
    # fig.colorbar(img3, ax=axs[3])

    plt.show()


# Para VAE
# Model
# model = model(
#     input_shape=input_shape,
#     conv_filters=conv_filters,
#     conv_kernels=conv_kernels,
#     conv_strides=conv_strides,
#     latent_space_dim=latent_space_dim,
#     output_padding=output_padding
# )
# model = model.load_checkpoint(63, 'model/test/load', test=True)

# Para UNET
from _CODE.unet_1 import *

input_shape = (input_shape[0]-1, *input_shape[1:])
model = build_unet(input_shape)
# model.compile(optimizer=Adam(lr=learning_rate), loss='mse', metrics=['mse'])
model.load_weights(os.path.join('model/test/load', f'cp-04.h5'))


def genearate_ids(path):
    ID_list_train = []
    path_train = os.path.join(path, 'train')
    for folder in os.listdir(path_train):
        for file_name in os.listdir(os.path.join(path_train, folder)):
            ID_list_train.append(os.path.join(folder, file_name))

    ID_list_validation = []
    path_validation = os.path.join(path, 'validation')
    for folder in os.listdir(path_validation):
        for file_name in os.listdir(os.path.join(path_validation, folder)):
            ID_list_validation.append(os.path.join(folder, file_name))

    return ID_list_train, ID_list_validation

# GENERATE AUDIO
metadata_file = [f for f in os.listdir(PATH_COMBINED) if f.endswith('.csv')]
with open(os.path.join(PATH_COMBINED, metadata_file[0]), 'r') as csvfile:
    df = pd.read_csv(csvfile)

idx = 100

# ID_list_train, ID_list_validation = genearate_ids(PATH_MELSPECTROGRAMS)
ID_list_train, ID_list_validation = genearate_ids(PATH_STFTS)

df = df[df['file_name'] == ID_list_train[idx].split('\\')[-1] + '.wav']
sound_name, sound_offset, duration, offset_noisy = \
    df[['sound_name', 'sound_offset', 'duration', 'noise_offset']].iloc[0]
audio_path = os.path.join(PATH_SONGS, sound_name)
noisy_audio_path = ID_list_train[idx]
generate_audio(model, audio_path, noisy_audio_path,
               offset_audio=sound_offset, offset_noisy=offset_noisy,
               duration=duration, play=True, save=True, type_data='spectogram',
               UNET=True)