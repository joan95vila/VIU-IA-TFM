from mylibs.audio.features.preprocess import *
from mylibs.ml.autoencoder import VAE

params_dict = {
    'segment_duration': 0.74,
    'sample_rate': 22050,

    'sound_start_time': 10,
    'noise_start_time': 60,
    'noise_end_time': 60,

    'sound_power': 0.5,
    'noise_power': 1,
    'max_songs': None,

    # STFT set up
    'frame_length': 1024 * 2,
    'overlap_percentage': 0.5,  # hop length
    'window': 'hann',

    # DATA GENERATOR set up
    'min_normalize_value': 0,
    'max_normalize_value': 1,
    'preprocessor': MinMaxNormaliser,
    'batch_size': 128,
    'shuffle': True,

    # MODEL setup
    'model': VAE,
    'input_shape': (128, 16, 1),
    # 'input_shape': (1025, 16, 1),
    # 'input_shape': (28, 28, 1), # MNIST
    'output_padding': [None, None, None, (1, 0), None],
    # 'output_padding': ((0, 1), (0, 1), (0, 1), 0, None),
    # 'output_padding': (None, 0, None, None, None),

    'conv_filters': (32, 64, 128, 256, 512),
    # 'conv_filters': (128, 513), # MNIST

    'conv_kernels': (3, 3, 3, 3, 3),
    'conv_strides': (2, 2, 2, 2, 2),
    'latent_space_dim': 128,
    # 'latent_space_dim': 532,
    # 'latent_space_dim': 16,

    # TRAIN set up
    'learning_rate': 1e-3,  # 2e-5, 2e-5, 1e-3  # (end, started)
    'epochs': 100,

    # LOSS set up
    'kl_loss_weight': 1,  # 0.001, 0.005, 0.0001 (started)
    'reconstruction_loss_weight': 1,

    # OTHERS
    'seed': 123,
    'max_samples': None,
    # 'max_samples': 2,
}

# NOTES
# + learning rate
#    - larger than 0.1e-2 create an exploding gradient in KL loss
