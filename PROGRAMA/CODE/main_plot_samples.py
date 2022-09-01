print("main_plot_samples.py has started.")

def predict_sample(model, x, y, i, y_type='cleaned', UNET=False):
    import matplotlib.pyplot as plt
    import librosa.display

    def find_dpi(w, h, d):
        """
        w : width in pixels
        h : height in pixels
        d : diagonal in inches
        """
        w_inches = (d ** 2 / (1 + h ** 2 / w ** 2)) ** 0.5
        return round(w / w_inches)

    dpi = find_dpi(3440, 1440, 34)
    # print(dpi)

    img_combined = np.expand_dims(x, axis=0)

    if UNET:
        reconstructed_image = model.predict(img_combined)
    else:
        reconstructed_image, latent_representation = model.reconstruct(img_combined)

    img_combined = np.squeeze(img_combined)
    reconstructed_image = np.squeeze(reconstructed_image)

    img_cleaned = np.squeeze(y)

    # print(reconstructed_image)
    # print(reconstructed_image.shape)
    # print(np.sum(np.isnan(reconstructed_image)))

    plt.figure(i)
    # fig, axs = plt.subplots(nrows=3, ncols=1, dpi=dpi, figsize=(76*0.39, 31.5*0.39))
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True,
                            dpi=dpi, figsize=(76 * 0.39 / 2, 31.5 * 0.39 / 2))

    axs[0].set_title('COMBINED')
    librosa.display.specshow(img_combined, x_axis='time', y_axis='linear', ax=axs[0],
                             n_fft=frame_length, hop_length=frame_length // 2)
    # librosa.display.specshow(img_combined, x_axis='time', y_axis='mel', ax=axs[0])

    axs[1].set_title(y_type.upper())
    img = librosa.display.specshow(img_cleaned, x_axis='time', y_axis='linear', ax=axs[1],
                                   n_fft=frame_length, hop_length=frame_length // 2)
    # img = librosa.display.specshow(img_cleaned, x_axis='time', y_axis='mel', ax=axs[1])

    axs[2].set_title('PREDICTION')
    librosa.display.specshow(reconstructed_image, x_axis='time', y_axis='linear', ax=axs[2],
                             n_fft=frame_length, hop_length=frame_length // 2)
    # librosa.display.specshow(reconstructed_image, x_axis='time', y_axis='mel', ax=axs[2])

    # fig.colorbar(img, ax=axs[2], orientation='horizontal')
    fig.colorbar(img, ax=axs)

    # https://colab.research.google.com/drive/19o9aU8wjuOao3cRrq0riy64Hk26uguFz?usp=sharing#scrollTo=Bu0FxwFARUvl

    # axs.autoscale(enable=True)

    def move_figure(f, x, y):
        import matplotlib

        """Move figure's upper left corner to pixel (x, y)"""
        backend = matplotlib.get_backend()
        # print(backend)
        if backend == 'TkAgg':
            f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
        elif backend == 'WXAgg':
            f.canvas.manager.window.SetPosition((x, y))
        else:
            # This works for QT and GTK
            # You can also use window.setGeometry
            f.canvas.manager.window.move(x, y)

    move_figure(fig, 0, 0)
    # plt.show()


def generate_sample(generator):
    X, Y = [], []
    for x, y in generator:
        for i in range(x.shape[0]):
            X.append(x[i])
            Y.append(y[i])
        return np.array(X), np.array(Y)



from configurations.params import *
from mylibs.audio.features.preprocess import *
from mylibs.ml.autoencoder import VAE
from mylibs.ml.train import *
from mylibs.utilities.datagen import *
import numpy as np
import os
import matplotlib.pyplot as plt


# PATH set up
path_manager = PathsManager(1)
PATH_COMBINED = path_manager.path_combined
PATH_ENV_AUDIOS = path_manager.path_env_audios
PATH_SONGS = path_manager.path_songs
PATH_STFTS = path_manager.path_stfts
PATH_MELSPECTROGRAMS = path_manager.path_melspectrograms

for key, val in params_dict.items():
    exec(key + '=val')


# Parameters
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

y_type = "cleaned"


# Generators
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

    return ID_list_train, ID_list_validation, path_train, path_validation

# GENERATE AUDIO
metadata_file = [f for f in os.listdir(PATH_COMBINED) if f.endswith('.csv')]
with open(os.path.join(PATH_COMBINED, metadata_file[0]), 'r') as csvfile:
    df = pd.read_csv(csvfile)

ID_list_train, ID_list_validation, path_train, path_validation = genearate_ids(PATH_STFTS)
# ID_list_train, ID_list_validation, path_train, path_validation = genearate_ids(PATH_MELSPECTROGRAMS)
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

# model = VAE.load_checkpoint(3, 'model/test/load', test=True,
#                             create_conf_file=True, params=parameters)

# PARA UNET
from _CODE.unet_1 import *

# input_shape = (input_shape[0], *input_shape[1:])
# model = build_unet(input_shape)
# model.compile(optimizer=Adam(lr=learning_rate), loss='mse', metrics=['mse'])
# model.load_weights(os.path.join('model/test/load', f'cp-05.h5'))

model = VAE.load_checkpoint(8, 'model/test/load', test=True,
                            create_conf_file=True, params=parameters)

generator_train = DataGenerator(path_train, np.array(ID_list_train),
                                y_type=y_type, **params)
generator_validation = DataGenerator(path_validation, np.array(ID_list_validation),
                                     y_type=y_type, **params)

# X, Y = generate_sample(generator_train)
X, Y = generate_sample(generator_validation)
max_samples = 8  # X.shape[0]
for i in range(max_samples):
    predict_sample(model, X[i], Y[i], i, y_type=y_type, UNET=False)

plt.show()

