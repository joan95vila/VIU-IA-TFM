import os
import pickle

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, LeakyReLU, BatchNormalization, \
    Flatten, Dense, Reshape, Conv2DTranspose, Activation, Lambda, Layer
from tensorflow.keras import backend as K
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()


class VAE:
    """
    VAE represents a Deep Convolutional variational autoencoder architecture
    with mirrored encoder and decoder components.
    """

    # https://blog.paperspace.com/how-to-build-variational-autoencoder-keras

    def __init__(self,
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim,
                 output_padding,
                 reconstruction_loss_weight=1,
                 kl_loss_weight=1):
        self.input_shape = input_shape  # [28, 28, 1]
        self.conv_filters = conv_filters  # [2, 4, 8]
        self.conv_kernels = conv_kernels  # [3, 5, 3]
        self.conv_strides = conv_strides  # [1, 2, 2]
        self.latent_space_dim = latent_space_dim  # 2
        self.output_padding = output_padding
        self.reconstruction_loss_weight = reconstruction_loss_weight  # 1000000
        self.kl_loss_weight = kl_loss_weight

        self.encoder = None
        self.decoder = None
        self.model = None

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None

        self._build()

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss=self._calculate_combined_loss,
                           metrics=[self._calculate_reconstruction_loss,
                                    self._calculate_kl_loss])
        # self.model.compile(optimizer=optimizer,
        #                    loss='bce',
        #                    metrics=['bce'])
        # self.model.compile(optimizer=optimizer,
        #                    loss='mse',
        #                    metrics=['mse'])

    def train(self, x_train, y_train, batch_size, epochs=1,
              callbacks=None, verbose=1, shuffle=True):
        self.model.fit(x_train,
                       y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       shuffle=shuffle,
                       callbacks=callbacks,
                       verbose=verbose)

    # def fit_generator(self, generator, validation_data, use_multiprocessing=True, workers=None):
    #     self.model.fit_generator(generator=generator,
    #                              validation_data=validation_data,
    #                              use_multiprocessing=use_multiprocessing,
    #                              workers=workers)

    def fit_generator(self, generator, validation_data, num_epochs=1, verbose=1, use_multiprocessing=True, workers=None,
                      callbacks=None):
        self.model.fit(generator,
                       validation_data=validation_data,
                       verbose=verbose,
                       epochs=num_epochs,
                       shuffle=True,
                       callbacks=callbacks,
                       # use_multiprocessing=use_multiprocessing,
                       # workers=workers
                       )

    def save(self, save_folder="."):
        self._create_folder_if_it_doesnt_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)
        self._save_model(save_folder)

    @staticmethod
    def _create_folder_if_it_doesnt_exist(folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim,
            self.output_padding,
            self.reconstruction_loss_weight,
            self.kl_loss_weight,
        ]
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights.h5")
        self.model.save_weights(save_path)

    def _save_model(self, save_folder):
        save_path = os.path.join(save_folder, "model.h5")
        self.model.save(save_path)

    @classmethod
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = VAE(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder.load_weights(weights_path)
        # for layer in autoencoder.layers: print(layer.get_config(), layer.get_weights())
        autoencoder.summary()
        print(Layer(name='mu').get_weights())
        print(Layer(name='log_variance').get_weights())
        print(Layer(name='decoder_input').get_weights())
        print(Layer(name='d').get_weights())
        print(Layer(name='encoder_leakyrelu_4').get_config())
        print(Layer(name='encoder_conv_layer_5').get_weights())
        print(Layer(name='encoder_bn_5').get_weights())

        return autoencoder

    @classmethod
    def load_checkpoint(cls, epoch, save_folder=".", test=False, create_conf_file=False, params=None):
        if create_conf_file:
            save_path = os.path.join(save_folder, "parameters.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(params, f)
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = VAE(*parameters)
        if test:
            # path = os.path.join(save_folder, f'cp-{epoch:02d}.h5')
            path = os.path.join(save_folder, f'{epoch}.h5')
        else:
            path = os.path.join(save_folder, 'tmp', f'cp-{epoch:02d}.h5')

        # checkpoint = tf.train.load_checkpoint(os.path.join(save_folder, 'tmp', 'checkpoint'))
        # checkpoint = tf.train.load_checkpoint(path)
        # checkpoint.restore(os.path.join(save_folder, 'tmp', 'checkpoint')).expect_partial()
        autoencoder.load_weights(path)
        # autoencoder.load_weights(os.path.join(save_folder, 'tmp', 'checkpoint'))
        # for layer in autoencoder.layers: print(layer.get_config(), layer.get_weights())
        autoencoder.summary()
        # print(Layer(name='mu').get_weights())
        # print(Layer(name='log_variance').get_weights())
        # print(Layer(name='decoder_input').get_weights())
        # print(Layer(name='decoder_bn_4/gamma:0').get_weights())
        # print(Layer(name='encoder_leakyrelu_4').get_config())
        # print(Layer(name='encoder_conv_layer_5').get_weights())
        # print(Layer(name='encoder_bn_5').get_weights())

        return autoencoder

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def reconstruct(self, images):
        latent_representations = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations

    # LOSS FUNCTIONS SECTION
    def _calculate_combined_loss(self, y_target, y_predicted):
        reconstruction_loss = self._calculate_reconstruction_loss(y_target, y_predicted)
        kl_loss = self._calculate_kl_loss()
        # combined_loss = self.reconstruction_loss_weight * reconstruction_loss \
        #                 + kl_loss*0  # + np.random.normal(0, 1, 1)[0]

        def count_decimals(d):
            d = str(d)[2:]
            count = 0
            for u in d:
                if u != '0':
                    return count
                count += 1
            return count

        # print('DSAF: ',
        #       reconstruction_loss[0], '\n',
        #       reconstruction_loss[1], '\n',
        #       reconstruction_loss[2], '\n',
        #       reconstruction_loss)

        if int(reconstruction_loss) == 0:
            print("rec0: ", -count_decimals(reconstruction_loss), int(reconstruction_loss))
            rec_magnitude = -count_decimals(reconstruction_loss)  # decimal part value
        else:
            print("rec-inf: ", len(str(int(reconstruction_loss))), int(reconstruction_loss))
            rec_magnitude = len(str(int(reconstruction_loss)))

        if int(kl_loss) == 0:
            print("kl0: ", -count_decimals(kl_loss), int(kl_loss))
            kl_magnitude = -count_decimals(kl_loss)
        else:
            print("kl-inf: ", len(str(int(kl_loss))), int(kl_loss))
            kl_magnitude = len(str(int(kl_loss)))

        diff_magnitude = rec_magnitude - kl_magnitude

        # print(rec_magnitude, kl_magnitude)

        rec_loss_weight, kl_loss_weight = 1, 1  # if same magnitude

        if rec_magnitude > 0 and kl_magnitude > 0:
            if diff_magnitude > 0:
                rec_loss_weight = 10 ** -diff_magnitude
            else:
                kl_loss_weight = 10 ** diff_magnitude

        elif rec_magnitude > 0 and kl_magnitude < 0:
            rec_loss_weight = 10 ** -diff_magnitude

        elif rec_magnitude < 0 and kl_magnitude > 0:
            kl_loss_weight = 10 ** -diff_magnitude

        elif rec_magnitude < 0 and kl_magnitude < 0:
            if diff_magnitude > 0:
                kl_loss_weight = 10 ** -diff_magnitude
            else:
                rec_loss_weight = 10 ** diff_magnitude

        rec_importance = 1
        kl_importance = 1

        # rec_loss_weight = 1
        # kl_loss_weight = 0.0001

        combined_loss = self.reconstruction_loss_weight * rec_importance * reconstruction_loss \
                        + self.kl_loss_weight * kl_importance * kl_loss

        return combined_loss

    def _calculate_reconstruction_loss(self, y_target, y_predicted):
        error = y_target - y_predicted
        reconstruction_loss = K.mean(K.square(error), axis=[1, 2, 3])
        # tf.print(reconstruction_loss, "Inside loss function 2")
        # reconstruction_loss = MSE(y_target, y_predicted)
        # print("SDSsDFDSFSDFSDFdsaf FDSFKSDKF sadfAS DKFKSDAF skadfSDAKf ksadFKS AKfsd")
        # print(reconstruction_loss)

        return K.clip(reconstruction_loss, 0, 1000)

    # def _calculate_kl_loss(self, y_target, y_predicted):
    def _calculate_kl_loss(self, *args):
        kl_loss = -0.5 * K.sum(1 + self.log_variance - K.square(self.mu) -
                               K.exp(self.log_variance), axis=1)

        return K.clip(kl_loss, 0, 1000)
        # return 0

    # AUTOENCODER SECTION
    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name="autoencoder")

    # ENCODER SECTION
    def _build_encoder(self):
        # x = Input(shape=(img_size, img_size, num_channels), name="encoder_input")
        # x = Input(shape=self.input_shape, name="encoder_input")

        encoder_input = self._add_encoder_input()

        conv_layers = self._add_conv_layers(encoder_input)

        bottleneck = self._add_bottleneck(conv_layers)

        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name="encoder")

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name="encoder_input")

    def _add_conv_layers(self, encoder_input):
        """Create all convolutional blocks in encoder."""
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x

    def _add_conv_layer(self, layer_index, x):
        """Add a convolutional block to a graph of layers, consisting of
        conv 2d + ReLU + batch normalization.
        """
        # layer_number = layer_index + 1
        layer_number = layer_index + 1
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"encoder_conv_layer_{layer_number}",
            data_format="channels_last"
        )
        x = conv_layer(x)
        x = BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
        x = LeakyReLU(name=f"encoder_leakyrelu_{layer_number}")(x)

        return x

    def _add_bottleneck(self, x):
        """Flatten data and add bottleneck with Gaussian sampling (Dense
        layer).
        """
        """
        The purpose of the shape_before_flatten variable is to hold the shape of 
        the result before being flattened, in order to decode the result 
        successfully.
        """
        self._shape_before_bottleneck = K.int_shape(x)[1:]
        x = Flatten()(x)

        KL = True
        if KL:
            self.mu = Dense(self.latent_space_dim, name="mu")(x)
            self.log_variance = Dense(self.latent_space_dim,
                                      name="log_variance")(x)

            # x = Dense(self.latent_space_dim, name="encoder_output")(x)

            def sample_point_from_normal_distribution(args):
                mu, log_variance = args
                epsilon = K.random_normal(shape=K.shape(mu), mean=0.,
                                          stddev=1.)
                sampled_point = mu + K.exp(log_variance / 2) * epsilon
                return sampled_point

            x = Lambda(sample_point_from_normal_distribution,
                       name="encoder_output")([self.mu, self.log_variance])
        else:
            x = Dense(self.latent_space_dim, name="latten")(x)

        return x

    # DECODER SECTION
    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)

        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)

        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

    def _add_decoder_input(self):
        return Input(shape=self.latent_space_dim, name="decoder_input")

    def _add_dense_layer(self, decoder_input):
        # numpy.prod() is used to multiply the three values and return just a single value.
        num_neurons = np.prod(self._shape_before_bottleneck)  # [1, 2, 4] -> 8
        dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)
        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        return Reshape(self._shape_before_bottleneck)(dense_layer)

    def _add_conv_transpose_layers(self, x):
        """Add conv transpose blocks."""
        # loop through all the conv layers in reverse order and stop at the
        # first layer
        for layer_index in reversed(range(1, self._num_conv_layers)):
            x = self._add_conv_transpose_layer(layer_index, x)
        return x

    def _add_conv_transpose_layer(self, layer_index, x):
        layer_num = self._num_conv_layers - layer_index
        layer_index -= 1
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            # filters=1,
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"decoder_conv_transpose_layer_{layer_num}",
            data_format="channels_last",
            output_padding=self.output_padding[layer_index]
        )
        x = conv_transpose_layer(x)
        x = BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
        x = LeakyReLU(name=f"decoder_leakyrelu_{layer_num}")(x)

        return x

    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters=1,
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers}",
            data_format="channels_last",
            output_padding=self.output_padding[0]
        )
        x = conv_transpose_layer(x)
        output_layer = Activation("sigmoid", name="sigmoid_layer")(x)
        # output_layer = Activation("linear", name="linear_layer")(x)
        return output_layer


if __name__ == "__main__":
    autoencoder = VAE(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2
    )
    autoencoder.summary()

    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
