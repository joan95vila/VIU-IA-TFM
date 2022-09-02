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

print("main_test_evaluation.py has started.")

# PATH set up
path_manager = PathsManager(1)
PATH_COMBINED = path_manager.path_combined
PATH_ENV_AUDIOS = path_manager.path_env_audios
PATH_SONGS = path_manager.path_songs
PATH_STFTS = path_manager.path_stfts
PATH_MELSPECTROGRAMS = path_manager.path_melspectrograms

PATH_STFTS_TEST = path_manager.path_stfts_test
PATH_COMBINED_TEST = path_manager.path_combined_test
PATH_NOISE_TEST = path_manager.path_env_audios
PATH_SONGS_TEST = path_manager.path_songs_test
PATH_MELESPECTROGRAM_TEST = path_manager.path_melspectrograms_test

# Parameters
for key, val in params_dict.items():
    exec(key + '=val')

parameters = [
    input_shape,
    conv_filters,
    conv_kernels,
    conv_strides,
    latent_space_dim,
    output_padding,
    reconstruction_loss_weight,
    kl_loss_weight,
]

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


# Generators
def genearate_ids(path, path_test):
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

    ID_list_test = []
    for folder in os.listdir(path_test):
        for file_name in os.listdir(os.path.join(path_test, folder)):
            ID_list_test.append(os.path.join(folder, file_name))

    return ID_list_train, ID_list_validation, ID_list_test, path_train, path_validation, path_test


ID_list_train, _, ID_list_test, _, _, path_test = genearate_ids(PATH_STFTS, PATH_STFTS_TEST)
# ID_list_train, _, ID_list_test, _, _, path_test = genearate_ids(PATH_MELSPECTROGRAMS, PATH_MELESPECTROGRAM_TEST)

# generator_train = DataGenerator(path_train, np.array(ID_list_train), **params)
# generator_validation = DataGenerator(path_validation, np.array(ID_list_validation), **params)
generator_test = DataGenerator(path_test, np.array(ID_list_test), **params)

CSVLogger_filepath = 'model/test/load/CSVLogger_test.log'
CSVLogger_filepath = 'D:\\__ MY __\\EDUCATION\\UNIVERSITIES\\VIU\\Master en Inteligencia Artificial\\10MIA - TFM\\__Devolupment (CODE)\\_CODE\\model\\test\\load\\CSVLogger_test.log'
model_CSVLogger_callback = CSVLogger(
    CSVLogger_filepath, separator=',', append=True
)
callbacks = [model_CSVLogger_callback]

# Para no UNET
model = VAE.load_checkpoint('VAE2', 'model/test/load', test=True,
                            create_conf_file=True, params=parameters)
model.compile(learning_rate)
results = model.model.evaluate(x=generator_test,
                     verbose=1,
                     use_multiprocessing=True,
                     workers=-1,
                     steps=None,
                     callbacks=callbacks)

# PARA UNET
# from _CODE.unet_1 import *
# # input_shape = (input_shape[0]-1, *input_shape[1:]) # No MEL
# input_shape = (input_shape[0], *input_shape[1:]) # MEL
# model = build_unet(input_shape)
# model.compile(optimizer=Adam(lr=learning_rate), loss='mse', metrics=['mse'])
# model.summary()
# model.load_weights(os.path.join('model/test/load', f'UNET1.h5'))
#
# results = model.evaluate(x=generator_test,
#                      verbose=1,
#                      use_multiprocessing=True,
#                      workers=-1,
#                      steps=None,
#                      callbacks=callbacks)

import json
with open('model/test/load/test_evaluation.json', 'w') as file:
    # json.dump(dict(zip(['Total loss', 'Reconstruction', 'KL'], map(str, results))), file, indent=0)
    json.dump(dict(zip(['Total loss'], map(str, results))), file, indent=0)

def generate_sample(generator):
    X, Y = [], []
    for x, y in generator:
        for i in range(x.shape[0]):
            X.append(x[i])
            Y.append(y[i])
        return np.array(X), np.array(Y)

X, Y = generate_sample(generator_test)
# print(X.shape)

print("DONE")