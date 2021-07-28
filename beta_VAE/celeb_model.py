import tensorflow as tf
class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim, beta=4, shape=[28,28,1]):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.beta = beta
    self.shape = shape
    self.output_f = int(shape[0]/4)
    self.output_s = shape[2]

    def _batch_norm(self, inputs):
        """Performs a batch normalization using a standard set of parameters."""
        # Set fused=True for a significant performance boost. See
        # https://www.tensorflow.org/performance/performance_guide#common_fused_ops

        return tf.compat.v1.layers.batch_normalization(
            inputs=inputs, axis=-1, momentum=Config.momentum, epsilon=Config.epsilon, center=True,
            scale=True, training=self.training, fused=True)

    def _conv_block(self, inputs, n_filters):

        w_init = tf.compat.v1.initializers.he_normal(seed=None)
        b_init = tf.constant_initializer(0.0)
        with tf.name_scope('conv_block'):
            conv = tf.compat.v1.layers.conv2d(inputs, filters=n_filters, kernel_size=(5, 5), strides=(2, 2),
                                              bias_initializer=b_init, kernel_initializer=w_init)
            batch_norm = self._batch_norm(conv)
            outputs = tf.nn.relu(batch_norm)
        return outputs

    def _deconv_block(self, inputs, n_filters):

        w_init = tf.compat.v1.initializers.he_normal(seed=None)
        b_init = tf.constant_initializer(0.0)
        with tf.name_scope('deconv_block'):
            deconv = tf.compat.v1.layers.conv2d_transpose(inputs, filters=n_filters, kernel_size=(5, 5),
                                                          strides=(2, 2), padding='same')
            batch_norm = self._batch_norm(deconv)
            outputs = tf.nn.relu(batch_norm)
        return outputs

    def encode(self, inputs):

        with tf.compat.v1.variable_scope('Encoder', reuse=tf.compat.v1.AUTO_REUSE):
            for i in range(len(self.filters)):
                inputs = self._conv_block(inputs, self.filters[i])
            flatten = tf.compat.v1.layers.flatten(inputs)
            dense1 = tf.compat.v1.layers.dense(flatten, units=self.latent_dim)
            z_mean = self._batch_norm(dense1)
            dense2 = tf.compat.v1.layers.dense(flatten, units=self.latent_dim)
            z_logvar = self._batch_norm(dense2)

        return z_mean, z_logvar

    def decode(self, inputs):

        with tf.compat.v1.variable_scope('Decoder', reuse=tf.compat.v1.AUTO_REUSE):
            inputs = tf.compat.v1.layers.dense(inputs, units=self.last_convdim * self.last_convdim * self.filters[-1])
            inputs = self._batch_norm(inputs)
            inputs = tf.reshape(inputs, [-1, self.last_convdim, self.last_convdim, self.filters[-1]])
            for i in range(len(self.filters) - 1, -1, -1):
                inputs = self._deconv_block(inputs, self.filters[i])
            output = tf.compat.v1.layers.conv2d_transpose(inputs, filters=3, kernel_size=(5, 5),
                                                          strides=(1, 1), padding='same')
            output = tf.nn.sigmoid(output)

        return output
