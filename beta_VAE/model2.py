import tensorflow as tf
import numpy as np
from math import floor
class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim, beta=4, shape=[28,28,1]):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.beta = beta
    self.shape = shape
    self.output_f = int(shape[0]/4)
    self.output_s = shape[2]
    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=shape),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=5, strides=(1, 1), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=5, strides=(2, 2), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(
                filters=128, kernel_size=5, strides=(2, 2), padding='same', use_bias=False),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim - 1 + latent_dim -1),
        ]
    )

    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=self.output_f * self.output_f *32, activation=tf.nn.relu),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Reshape(target_shape=(self.output_f, self.output_f, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=128, kernel_size=5, strides=1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(
                filters=self.output_s, kernel_size=3, strides=1, padding='same'),
        ]
    )

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits

class Classifier(tf.keras.Model):
    def __init__(self, shape, num_cls):
        super(Classifier, self).__init__()
        self.shape = shape
        self.num_cls = num_cls
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.shape)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(1, 1), activation='relu'),
                tf.keras.layers.MaxPool2D(2,2),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(1, 1), activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                # No activation
                tf.keras.layers.Dense(10),
            ]
        )

    def call(self, X):
        return self.model(X)



