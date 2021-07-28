import tensorflow as tf
import numpy as np
from math import floor
class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim, beta=4, shape=[28,28,1], model="cnn"):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.beta = beta
    self.shape = shape
    self.angle_dim = 2
    self.output_f = int(shape[0]/4)
    self.output_s = shape[2]
    if (model == "cnn"):
        self.encoder = tf.keras.Sequential(
            [
            tf.keras.layers.InputLayer(input_shape=shape),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim+1,)),
                tf.keras.layers.Dense(units=self.output_f * self.output_f *32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(self.output_f, self.output_f, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=self.output_s, kernel_size=3, strides=1, padding='same'),
            ]
        )
    elif (model == "mlp"):
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=shape),
                tf.keras.layers.Dense(
                    64, activation='relu'),
                tf.keras.layers.Dense(
                32, activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim+1,)),
                tf.keras.layers.Dense(latent_dim * latent_dim, activation=tf.nn.relu),
                tf.keras.layers.Dense(
                    512, activation='relu'),
                tf.keras.layers.Dense(
                    2352,
                    activation='relu'),
                # No activation
                tf.keras.layers.Reshape(target_shape=[28,28,3]),
                tf.keras.layers.Dense(
                    1)
            ]
        )

        assert self.decoder.output_shape == (None, 28, 28, 1)

  @tf.function
  def sample(self, degree, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(degree, eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    Y = eps * tf.exp(logvar * .5) + mean
    return Y

  def split_identity(self, mean, logvar):
      return self.reparameterize(mean, logvar, id=True)

  def decode(self, Z, Y, apply_sigmoid=False):
    degree_matrix = tf.cast(tf.fill([Y.shape[0],1], Z), tf.float32)
    input = tf.concat([degree_matrix, Y], 1)
    logits = self.decoder(input)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits


class Z_Encoder(tf.keras.Model):
    def __init__ (self, x_size=[28,28,1], factor_dims=2):
        super(Z_Encoder, self).__init__()
        self.input_size = x_size
        self.factor_dims = factor_dims
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=x_size),
                tf.keras.layers.Dense(
                    64, activation='relu'),
                tf.keras.layers.Dense(
                32, activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(factor_dims)
            ]
        )

    def sample(self, x, factor):
        return self.decode(x, factor, apply_sigmoid=True)