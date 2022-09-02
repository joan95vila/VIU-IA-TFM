from mylibs.ml.autoencoder import VAE
from configurations.params import *
import visualkeras
import tensorflow as tf

tf.executing_eagerly()

for key, val in params_dict.items():
    exec(key + '=val')

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

model = model.load_checkpoint(38, save_folder='model', test=True)

# visualkeras.graph_view(model)
tf.keras.utils.plot_model(model)
