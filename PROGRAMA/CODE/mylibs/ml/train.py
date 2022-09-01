from mylibs.ml.autoencoder import VAE
from mylibs.utilities.datagen import *
from configurations.paths import *


# https://blog.keras.io/building-autoencoders-in-keras.html
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

# # Guardamos el modelo
# model.save(BASE_FOLDER + 'models/' + TRAINING_NAME + ".h5")
#
# # Guardamos el historial del entrenamiento y la persición final del modelo
# with open(BASE_FOLDER + 'models/' + TRAINING_NAME + '.H', 'wb') as file:
#     pickle.dump(H_aug, file)


def train(model, x_train, y_train, learning_rate=0.0005, batch_size=32, epochs=1,
          callbacks=None, verbose=1):
    autoencoder = model
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train,
                      y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      shuffle=True,
                      callbacks=callbacks,
                      verbose=verbose
                      )

    return autoencoder


def fit_generator(model, training_generator, validation_generator, learning_rate=0.0005,
                  epochs=1, verbose=1, use_multiprocessing=True, workers=None,
                  callbacks=None):
    autoencoder = model
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.fit_generator(
        generator=training_generator,
        validation_data=validation_generator,
        num_epochs=epochs,
        verbose=verbose,
        use_multiprocessing=True,
        workers=-1,
        callbacks=callbacks
    )

    return autoencoder


if __name__ == "__main__":
    from configurations.paths import *
    from mylibs.audio.features.preprocess import *
    import numpy as np
    import os

    path_manager = PathsManager(2)
    PATH_COMBINED = path_manager.path_combined
    PATH_SONGS = path_manager.path_songs
    PATH_STFTS = path_manager.path_stfts

    import nvsmi
    import subprocess, pprint
    pprint.pprint([i for i in nvsmi.get_gpus()])
    # pprint.pprint([i for i in nvsmi.get_available_gpus()])
    # pprint.pprint([i for i in nvsmi.get_gpu_processes()])

    np_object = np.load(os.path.join(PATH_STFTS, os.listdir(PATH_STFTS)[1], 'combined.npy'))
    np_object = np.expand_dims(np_object, axis=2)
    dim = np_object.shape

    preprocessor = MinMaxNormaliser(1, 0)

    # Parameters
    params = {
        'dim': np.squeeze(np_object).shape,
        # 'dim': [64, 64],  # (1025, 130),
        'batch_size': 32,  # 64
        'n_channels': 1,
        'shuffle': True,
        'preprocessor': preprocessor,
        'sr': 22050,
        'n_fft': 1024*2,
        'hop_length': 1024,
        'window': 'hann'
    }

    learning_rate = 0.0005
    epochs = 2

    # Datasets
    ID_list = [file for file in os.listdir(PATH_STFTS) if not file.endswith('.py')]

    # Generators
    generator = DataGenerator(os.path.join(PATH_STFTS), ID_list, **params)

    # training_generator = DataGenerator(partition['train'], labels, **params)
    # validation_generator = DataGenerator(partition['validation'], labels, **params)

    # Design model
    # autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    model = VAE(
        input_shape=dim,
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, (2, 1)),
        latent_space_dim=128,
        output_padding=((0, 1), (0, 1), (0, 1), 0, None)
    )

    autoencoder = fit_generator(model, generator, generator, learning_rate=learning_rate, epochs=epochs)

    autoencoder.save("model")
