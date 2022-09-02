from mylibs.audio.features.stft import *
from mylibs.audio.mix.overlay import *
from configurations.paths import *
from configurations.params import *
from mylibs.audio.features.preprocess import *
from mylibs.ml.autoencoder import VAE
from mylibs.ml.train import *
from mylibs.utilities.datagen import *
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
import numpy as np
import os
import tensorflow as tf

# tf.compat.v1.enable_eager_execution()

OVERLAY = False
STFTS = False
MELSPECTROGRAMS = False
CREATE_DATASET = False
TRAIN = True

# PATH set up
path_manager = PathsManager(1)
PATH_COMBINED = path_manager.path_combined
PATH_ENV_AUDIOS = path_manager.path_env_audios
PATH_SONGS = path_manager.path_songs
PATH_STFTS = path_manager.path_stfts
PATH_MELSPECTROGRAMS = path_manager.path_melspectrograms

for key, val in params_dict.items():
    exec(key + '=val')

if OVERLAY:
    print("Overlay stage has started.")

    overlay = Overlay(PATH_SONGS, PATH_ENV_AUDIOS, PATH_COMBINED, segment_duration,
                      sound_start_time=sound_start_time,
                      noise_start_time=noise_start_time, noise_end_time=noise_end_time
                      )  # initialize object
    overlay.overlay_by_folders_sequential(sound_power=sound_power, noise_power=noise_power,
                                          max_songs=max_songs)  # overlay

if STFTS:
    print("STFT's stage has started.")

    audio_features = AudioFeatures()  # initialize object
    audio_features.extract_combined_audios_stfts(PATH_COMBINED, PATH_SONGS, PATH_STFTS,
                                                 n_fft=frame_length,
                                                 hop_length=int(frame_length * overlap_percentage))
    # save and calculation
if MELSPECTROGRAMS:
    print("Melspectrograms stage has started.")

    ID_list_aux1 = pd.read_csv(os.path.join(PATH_COMBINED, 'metadata.csv'))['file_name'].map(
        lambda x: ''.join(x.split('.')[:-3]))
    ID_list_aux2 = pd.read_csv(os.path.join(PATH_COMBINED, 'metadata.csv'))['file_name'].map(lambda x: x[:-4])
    ID_list = list(map(lambda f1, f2: os.path.join(f1, f2), ID_list_aux1, ID_list_aux2))

    path_melspectograms_train = os.path.join(PATH_MELSPECTROGRAMS, 'train')

    audio_features = AudioFeatures()
    audio_features.extract_combined_audios_stfts_from_stft(PATH_STFTS, path_melspectograms_train,
                                                           ID_list,
                                                           preprocessor=preprocessor(0, 1),
                                                           n_fft=frame_length,
                                                           hop_length=int(frame_length * overlap_percentage),
                                                           sr=sample_rate)

if CREATE_DATASET:
    print("Create dataset stage has started.")

    audio_features = AudioFeatures()
    audio_features.create_validation_dataset(PATH_MELSPECTROGRAMS, percentage=0.2)

if TRAIN:
    print("Train stage has started.")


    def predict_sample(model, x, y, i, y_type='cleaned'):
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
        reconstructed_image, latent_representation = model.reconstruct(img_combined)
        img_combined = np.squeeze(img_combined)
        reconstructed_image = np.squeeze(reconstructed_image)

        img_cleaned = np.squeeze(y)

        # print(reconstructed_image)
        # print(reconstructed_image.shape)
        # print(np.sum(np.isnan(reconstructed_image)))

        plt.figure(i)
        # fig, axs = plt.subplots(nrows=3, ncols=1, dpi=dpi, figsize=(76*0.39, 31.5*0.39))
        fig, axs = plt.subplots(nrows=3, ncols=1, dpi=dpi, figsize=(76 * 0.39 / 2, 31.5 * 0.39 / 2))

        axs[0].set_title('COMBINED')
        librosa.display.specshow(img_combined, x_axis='time', y_axis='mel', ax=axs[0],
                                 n_fft=frame_length, hop_length=frame_length // 2)

        axs[1].set_title(y_type.upper())
        img = librosa.display.specshow(img_cleaned, x_axis='time', y_axis='mel', ax=axs[1],
                                       n_fft=frame_length, hop_length=frame_length // 2)

        axs[2].set_title('PREDICTION')
        librosa.display.specshow(reconstructed_image, x_axis='time', y_axis='mel', ax=axs[2],
                                 n_fft=frame_length, hop_length=frame_length // 2)

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


    def plot_generator(generator, y_type='cleaned'):
        for x, y in generator:
            import matplotlib.pyplot as plt

            idx = 4
            print("shapes --> x, y: ", x.shape, y.shape)
            print("types  --> x, y: ", x.dtype, y.dtype)
            print("values --> x: ", x)
            x, y = np.squeeze(x[idx]), np.squeeze(y[idx])

            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
            axs[0].set_title('COMBINED')
            librosa.display.specshow(x, x_axis='time', y_axis='mel', ax=axs[0],
                                     n_fft=frame_length, hop_length=frame_length // 2)
            axs[1].set_title(y_type.upper())
            img = librosa.display.specshow(y, x_axis='time', y_axis='mel', ax=axs[1],
                                           n_fft=frame_length, hop_length=frame_length // 2)
            fig.colorbar(img, ax=axs[1])
            # https://colab.research.google.com/drive/19o9aU8wjuOao3cRrq0riy64Hk26uguFz?usp=sharing#scrollTo=Bu0FxwFARUvl

            plt.show()

            break


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

    # with open(os.path.join('model', 'model_params.pkl'), 'wb+') as file:
    #     import pickle

        # pickle.dump (params, file)

    # Callbacks
    # checkpoint_filepath = 'model\\tmp\\checkpoint.epoch-{epoch}'
    checkpoint_filepath = 'model\\tmp\\cp-{epoch:02d}.h5'
    model_checkpoint_callback = ModelCheckpoint(
        checkpoint_filepath,
        monitor='val_loss',
        verbose=0,
        save_best_only=False,
        save_format='h5',
        save_weights_only=False,
        mode='auto',
        save_freq=1,  # 'epoch',
        options=None,
        initial_value_threshold=None
    )

    CSVLogger_filepath = 'model/tmp/CSVLogger.log'
    # CSVLogger.on_test_begin = CSVLogger.on_train_begin
    # CSVLogger.on_test_batch_end = CSVLogger.on_epoch_end
    # CSVLogger.on_test_end = CSVLogger.on_train_end
    model_CSVLogger_callback = CSVLogger(
        CSVLogger_filepath, separator=',', append=True
    )


    class LossHistory(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.losses = []

        # def on_batch_end(self, batch, logs={}):
        #     self.losses.append(logs.get('loss'))

        # https://stackoverflow.com/a/53653154/852795
        def on_epoch_end(self, epoch, logs=None):
            # https://stackoverflow.com/a/54092401/852795
            import json, codecs

            def saveHist(path, history):
                with codecs.open(path, 'w', encoding='utf-8') as f:
                    json.dump(history, f, separators=(',', ':'), sort_keys=True, indent=4)

            def loadHist(path):
                n = {}  # set history to empty
                if os.path.exists(path):  # reload history if it exists
                    with codecs.open(path, 'r', encoding='utf-8') as f:
                        n = json.loads(f.read())
                return n

            def appendHist(h1, h2):
                if h1 == {}:
                    return h2
                else:
                    dest = {}
                    # print(h1)
                    for key, value in h1.items():
                        dest[key] = value + h2[key]
                    return dest

            history_filename = 'model-history.json'
            new_history = {}
            print()
            print("A: ", new_history)
            print("B: ", logs)
            for k, v in logs.items():  # compile new history from logs
                new_history[k] = [v]  # convert values into lists
            current_history = loadHist(history_filename)  # load history from current training
            current_history = appendHist(current_history, new_history)  # append the logs
            saveHist(history_filename, current_history)  # save history from current training

    reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=2, min_lr=learning_rate/10000, verbose=1, min_delta=1e-4)


    # class LossHistory(tf.keras.callbacks.Callback):
    #     def on_train_begin(self, logs={}):
    #         if not os.path.exists('tmp/mylog.log'):
    #             header = ['']
    #         with open("tmp/mylog.log", mode='a+') as file:
    #             csvwriter = csv.writer(file)  # 2. create a csvwriter object
    #             # csvwriter.writerow(header)  # 4. write the header
    #             csvwriter.writerow(logs)  # 4. write the header
    #             csvwriter.writerow(self.losses)  # 4. write the header
    #             # csvwriter.writerows(data)  # 5. write the rest of the data
    #         self.losses = []
    #
    #     def on_batch_end(self, batch, logs={}):
    #         self.losses.append(logs.get('loss'))


    # https://stackoverflow.com/questions/56388324/csvlogger-does-not-work-for-model-evaluate-process-for-keras


    # print(history.losses) # call after fit_model
    # outputs
    '''
    [0.66047596406559383, 0.3547245744908703, ..., 0.25953155204159617, 0.25901699725311789]
    '''


    # history_callback = LossHistory()
    history_checkpoint = LossHistory()
    # callbacks = [model_checkpoint_callback, model_CSVLogger_callback, reduce_lr_callback, history_checkpoint] # history_callback
    callbacks = [model_checkpoint_callback, model_CSVLogger_callback, reduce_lr_callback] # history_callback

    # Training
    model = model(
        input_shape=input_shape,
        conv_filters=conv_filters,
        conv_kernels=conv_kernels,
        conv_strides=conv_strides,
        latent_space_dim=latent_space_dim,
        output_padding=output_padding,
        reconstruction_loss_weight=reconstruction_loss_weight,
        kl_loss_weight=kl_loss_weight,
    )

    # Generators
    ID_list_train = []
    mode = 'mel'
    # mode = 'stft'
    # mode = 'spectogram'
    # y_type = 'noise'
    # y_type = 'combined'
    y_type = 'cleaned'
    if max_samples:
        max_samples = max_samples * batch_size
        A = list(range(0, 6000))
        np.random.seed(123)
        np.random.shuffle(A)
        A = A[:max_samples]

    if mode == 'mel':
        path_melspectograms_train = os.path.join(PATH_MELSPECTROGRAMS, 'train')
        for folder in os.listdir(path_melspectograms_train):
            for file_name in os.listdir(os.path.join(path_melspectograms_train, folder)):
                ID_list_train.append(os.path.join(folder, file_name))

        ID_list_validation = []
        path_melspectograms_validation = os.path.join(PATH_MELSPECTROGRAMS, 'validation')
        for folder in os.listdir(path_melspectograms_validation):
            for file_name in os.listdir(os.path.join(path_melspectograms_validation, folder)):
                ID_list_validation.append(os.path.join(folder, file_name))

        if max_samples:
            generator_train = DataGenerator(path_melspectograms_train, np.array(ID_list_train)[A], y_type=y_type, **params)
            generator_validation = DataGenerator(path_melspectograms_validation, np.array(ID_list_validation)[A], y_type=y_type,
                                                 **params)
        else:
            generator_train = DataGenerator(path_melspectograms_train, np.array(ID_list_train), y_type=y_type, **params)
            generator_validation = DataGenerator(path_melspectograms_validation, np.array(ID_list_validation), y_type=y_type, **params)

    elif mode == 'spectogram':
        path_train = os.path.join(PATH_STFTS, 'train')
        for folder in os.listdir(path_train):
            for file_name in os.listdir(os.path.join(path_train, folder)):
                ID_list_train.append(os.path.join(folder, file_name))

        ID_list_validation = []
        path_validation = os.path.join(PATH_STFTS, 'validation')
        for folder in os.listdir(path_validation):
            for file_name in os.listdir(os.path.join(path_validation, folder)):
                ID_list_validation.append(os.path.join(folder, file_name))

        if max_samples:
            generator_train = DataGenerator(path_train, np.array(ID_list_train)[A], y_type=y_type, **params)
            generator_validation = DataGenerator(path_validation, np.array(ID_list_validation)[A], y_type=y_type,
                                                 **params)
        else:
            generator_train = DataGenerator(path_train, np.array(ID_list_train), y_type=y_type, **params)
            generator_validation = DataGenerator(path_validation, np.array(ID_list_validation), y_type=y_type, **params)
        # generator_validation = generator_train

    elif mode == 'stft':
        for folder in os.listdir(PATH_STFTS)[1:]:
            for file_name in os.listdir(os.path.join(PATH_STFTS, folder)):
                ID_list_train.append(os.path.join(folder, file_name))
        ID_list_train = ID_list_train

        generator_train = DataGenerator(PATH_STFTS, ID_list_train[:max_samples], **params)
        generator_validation = generate_sample

    # plot_generator(generator_train)

    # Model
    TRAIN = True
    LOAD = not TRAIN
    # TRAIN = False
    LOAD = True
    MNIST = False

    if max_samples:
        pass
        # print(np.array(ID_list_train)[A])
    # ['[cafeteria]-Angela Thomas Wade - Milk Cow Blues\\[cafeteria]-Angela Thomas Wade - Milk Cow Blues.stem.168'
    #  '[cafeteria]-AM Contra - Heart Peripheral\\[cafeteria]-AM Contra - Heart Peripheral.stem.155']

    if LOAD and not MNIST:
        # model = model.load('model')
        # model = model.load_checkpoint(38, save_folder='model', test=True)

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

        import pickle

        save_path = os.path.join('model', "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

        model = model.load_checkpoint(10, 'model', test=True)

    if not TRAIN and not MNIST:
        import matplotlib.pyplot as plt

        # idx = np.random.randint(0, batch_size)
        idx = 0

        X, Y = generate_sample(generator_train)
        for i in range(X.shape[0]):
            predict_sample(model, X[i], Y[i], i, y_type=y_type)

        plt.show()

    elif not MNIST:
        import json

        with open("model/params_dict.json", "w") as file:
            params_dict_mod = params_dict
            params_dict_mod['preprocessor'] = 'MinMax'
            params_dict_mod['model'] = 'VAE'
            json.dump(params_dict_mod, file, indent="")

        autoencoder = fit_generator(model, generator_train, generator_validation,
                                    learning_rate=learning_rate, epochs=epochs,
                                    callbacks=callbacks, verbose=1)

        idx = 0
        x, y = generate_sample(generator_train)
        predict_sample(model, x[idx], y[idx], 0, y_type=y_type)

        autoencoder.save("model")

    if MNIST:
        import matplotlib.pyplot as plt

        TRAIN = True

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        np.random.shuffle(x_train)
        x_train = x_train[:10000]
        x_train = np.expand_dims(x_train, axis=3)
        print(x_train.shape)

        if TRAIN:
            autoencoder = train(model,
                                x_train,
                                x_train,
                                learning_rate=learning_rate,
                                epochs=epochs,
                                callbacks=callbacks,
                                verbose=1)
            autoencoder.save("model")  # creates teh file parameters.pkl

        # model = model.load('model')
        model = model.load_checkpoint(63, 'model')

        reconstructed_image, latent_representation = model.reconstruct(x_train)
        print(latent_representation.shape)
        reconstructed_image = reconstructed_image
        reconstructed_image = np.squeeze(reconstructed_image)
        for i in range(4):
            plt.figure(i)
            plt.imshow(reconstructed_image[i])
        plt.show()