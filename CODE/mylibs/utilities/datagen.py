import numpy as np
from tensorflow.keras.utils import Sequence
import os
import librosa


class DataGenerator(Sequence):
    """Generates data for Keras"""

    # https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    def __init__(self, dataset_path, list_IDs, preprocessor=None, batch_size=32, dim=(32, 32, 32), n_channels=1,
                 n_classes=10, shuffle=True, seed=None, y_type='cleaned',
                 sr=22050, n_fft=1024, hop_length=1028 // 2, window='hann', n_mels=128):
        """Initialization"""
        self.indexes = np.arange(len(list_IDs))
        self.dim = dim
        self.batch_size = batch_size
        # self.labels = labels
        self.dataset_path = dataset_path
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle

        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window
        self.n_mels = n_mels

        self.y_type = y_type

        self.preprocessor = preprocessor

        np.random.seed(seed)

        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        # X, Y, X_prime, Y_prime = self.__data_generation(list_IDs_temp)
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y  # , X_prime, Y_prime

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""

        # Initialization
        # Comment this line if no UNET model is used.
        # X = np.empty((self.batch_size, self.dim[0]-1, self.dim[1], self.n_channels))
        # Discomment this line if no UNET model is used.
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty_like(X)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # load sample
            x = np.load(os.path.join(self.dataset_path, ID, 'combined.npy'))
            y = np.load(os.path.join(self.dataset_path, ID, f'{self.y_type}.npy'))

            # Comment this block if no UNET model is used.
            # x = x[:-1, ...]
            # y = y[:-1, ...]
            # print(x.shape)

            X[i] = x
            Y[i] = y

            # print("data info [DATA GENERATOR] --> x, y\n"
            #       "\t(min, max, mean)\n"
            #       '\tx: ', np.min(x), np.max(x), np.mean(x), '\n'
            #       '\ty: ', np.min(y), np.max(y), np.mean(y))

        return X, Y

    def __data_generation_bakcup(self, list_IDs_temp):
        """Generates data containing batch_size samples"""

        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty_like(X)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            x = np.load(os.path.join(self.dataset_path, ID, 'combined.npy'))
            y = np.load(os.path.join(self.dataset_path, ID, f'{self.y_type}.npy'))

            x, y = np.abs(x), np.abs(y)
            x, y = librosa.amplitude_to_db(x), librosa.amplitude_to_db(y)

            # print("shapes --> x, y: ", x.shape, y.shape)
            # print("types  --> x, y: ", x.dtype, y.dtype)
            # print("values --> x: ", x)

            X[i] = np.expand_dims(x, axis=2)
            Y[i] = np.expand_dims(y, axis=2)

        return X, Y


if __name__ == '__main__':
    from configurations.paths import *
    from mylibs.audio.features.preprocess import *

    path_manager = PathsManager(2)
    PATH_COMBINED = path_manager.path_combined
    PATH_SONGS = path_manager.path_songs
    PATH_STFTS = path_manager.path_stfts
    PATH_MELSPECTROGRAMS = path_manager.path_melspectrograms

    print("[MAIN datagen] has started")


    def plot_data_generator(S_X_mel, S_Y_mel, n_fft=1024 * 2, hop_length=1024):
        import matplotlib.pyplot as plt

        S_X_mel = np.squeeze(S_X_mel)
        S_Y_mel = np.squeeze(S_Y_mel)

        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=False)

        ax[0].set_title('Spectrogram mel [COMBINED]')
        librosa.display.specshow(S_X_mel, x_axis='time', y_axis='mel', ax=ax[0], n_fft=n_fft,
                                 hop_length=hop_length)
        ax[1].set_title('Spectrogram mel [CLEANED]')
        img = librosa.display.specshow(S_Y_mel, x_axis='time', y_axis='mel', ax=ax[1], n_fft=n_fft,
                                       hop_length=hop_length)
        fig.colorbar(img, ax=ax)
        plt.show()


    ID_list_train = []
    # path_melspectograms_train = os.path.join(PATH_MELSPECTROGRAMS, 'train')
    # for folder in os.listdir(path_melspectograms_train):
    #     for file_name in os.listdir(os.path.join(path_melspectograms_train, folder)):
    #         ID_list_train.append(os.path.join(folder, file_name))

    for folder in os.listdir(PATH_STFTS)[1:]:
        for file_name in os.listdir(os.path.join(PATH_STFTS, folder)):
            ID_list_train.append(os.path.join(folder, file_name))
    ID_list_train = ID_list_train

    # generator = DataGenerator(os.path.join(PATH_ROOT, PATH_STFTS), ID_list[:100], dim=(1025, 130))
    preprocessor = MinMaxNormaliser(1, 0)
    # generator_train = DataGenerator(path_melspectograms_train, ID_list_train, preprocessor=preprocessor,
    #                           dim=(1025, 16), sr=22050, n_fft=1024*2, hop_length=1024, window='hann')
    generator_train = DataGenerator(PATH_STFTS, ID_list_train, preprocessor=preprocessor, seed=1234,
                                    dim=(1025, 16), sr=22050, n_fft=1024 * 2, hop_length=1024, window='hann')

    data = []  # store all the generated data batches
    labels = []  # store all the generated label batches
    max_iter = 20000  # maximum number of iterations, in each iteration one batch is generated; the proper value depends
    # on batch size and size of whole data
    i = 0
    for x, y in generator_train:
        # print(x.shape, y.shape)
        # print(sys.getsizeof(np.stack((x, y))))
        # print(sys.getsizeof(y[0].copy()), sys.getsizeof(x)//32, sys.getsizeof(x), len(str(sys.getsizeof(x))))

        data.append(x)
        labels.append(y)

        idx = 5
        # plot_data_generator(x[idx], y[idx], n_fft=1024*2, hop_length=1024)
        # break

        # print('x ', np.min(x), np.max(x), np.round(np.mean(x), 2))
        # print('y ', np.min(y), np.max(y), np.round(np.mean(y), 2))

        i += 1
        if i == max_iter:
            break

    # print(np.shape(data))
