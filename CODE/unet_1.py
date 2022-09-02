# Building Unet by dividing encoder and decoder into blocks

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Activation, MaxPool2D, Concatenate, \
    Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.optimizers import Adam


# from tensorflow.layers import Activation, MaxPool2D, Concatenate


def conv_block(input, num_filters, strides=(1, 1)):
    x = Conv2D(num_filters, 3, strides=strides, padding="same")(input)
    x = BatchNormalization()(x)  # Not in the original network.
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, strides=strides, padding="same")(x)
    x = BatchNormalization()(x)  # Not in the original network
    x = Activation("relu")(x)

    return x


# Encoder block: Conv block followed by maxpooling


def encoder_block(input, num_filters, strides=(1, 1)):
    x = conv_block(input, num_filters, strides=strides)
    # p = MaxPool2D((1, 2), strides=(2, 2))(x)
    p = MaxPool2D((2, 2))(x)
    # p = Dropout(0.5)(p)

    return x, p


# Decoder block
# skip features gets input from encoder for concatenation

def decoder_block(input, skip_features, num_filters, strides=2, padding="same", output_padding=None):
    x = Conv2DTranspose(num_filters, (2, 2), strides=strides, padding=padding, output_padding=output_padding)(input)
    x = Concatenate()([x, skip_features])
    x = Dropout(0.3)(x)
    x = conv_block(x, num_filters)
    return x


# Build Unet using the blocks
def build_unet(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024, strides=(1, 1))  # Bridge

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)  # Binary (can be multiclass)

    model = Model(inputs, outputs, name="U-Net")

    return model

if __name__ == '__main__':
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

    for key, val in params_dict.items():
        exec(key + '=val')

    # PATH set up
    path_manager = PathsManager(1)
    PATH_COMBINED = path_manager.path_combined
    PATH_ENV_AUDIOS = path_manager.path_env_audios
    PATH_SONGS = path_manager.path_songs
    PATH_STFTS = path_manager.path_stfts
    PATH_MELSPECTROGRAMS = path_manager.path_melspectrograms

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

    reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=2, min_lr=learning_rate/10000, verbose=1, min_delta=1e-4)

    callbacks = [model_checkpoint_callback, model_CSVLogger_callback, reduce_lr_callback]

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

    # ID_list_train, ID_list_validation, path_train, path_validation = genearate_ids(PATH_STFTS)
    ID_list_train, ID_list_validation, path_train, path_validation = genearate_ids(PATH_MELSPECTROGRAMS)

    y_type = 'cleaned'

    if max_samples:
        max_samples = max_samples * batch_size
        A = list(range(0, 6000))
        np.random.seed(123)
        np.random.shuffle(A)
        A = A[:max_samples]

        generator_train = DataGenerator(path_train, np.array(ID_list_train)[A], y_type=y_type, **params)
        generator_validation = DataGenerator(path_validation, np.array(ID_list_validation)[A], y_type=y_type,
                                             **params)
    else:
        generator_train = DataGenerator(path_train, np.array(ID_list_train), y_type=y_type, **params)
        generator_validation = DataGenerator(path_validation, np.array(ID_list_validation), y_type=y_type, **params)


    # input_shape = (input_shape[0] - 1, *input_shape[1:])
    input_shape = (input_shape[0], *input_shape[1:]) # datagen.py has also to be modified
    model = build_unet(input_shape)

    LOAD = True
    if LOAD:
        epoch = 2
        path = 'D:\\__ MY __\\EDUCATION\\UNIVERSITIES\\VIU\\Master en Inteligencia Artificial\\10MIA - TFM\\__Devolupment (CODE)\\_CODE'
        path = os.path.join(path, 'model', 'tmp', f'cp-{epoch:02d}.h5')

        model.load_weights(path)

    # model.compile(optimizer=Adam(lr = 1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=Adam(lr=learning_rate), loss='mse', metrics=['mse'])
    model.summary()
    #
    # history = model.fit_generator(my_generator, validation_data=validation_datagen,
    #                     steps_per_epoch=steps_per_epoch,
    #                     validation_steps=steps_per_epoch, epochs=25)

    model.fit(generator_train,
              validation_data=generator_validation,
              verbose=1,
              epochs=10,
              shuffle=True,
              callbacks=callbacks,
              use_multiprocessing=False,
              workers=-1
              )
