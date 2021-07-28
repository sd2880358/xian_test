import tensorflow as tf
from model import CVAE
from dataset import preprocess_images, divide_dataset
import numpy as np
from tensorflow_addons.image import rotate
from loss import *


class Merge_VAE(tf.keras.Model):
    def __init__(self, latent_dim, encoder, decoder, beta=4, shape=[28, 28, 1]):
        super(Merge_VAE, self).__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.shape = shape
        self.angle_dim = 2
        self.output_f = int(shape[0] / 4)
        self.output_s = shape[2]
        self.encoder = encoder
        self.decoder = decoder

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar, id=False):
        eps = tf.random.normal(shape=mean.shape)
        z = eps * tf.exp(logvar * .5) + mean
        return z

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

def merge_network(teacher, student):
    # import layers/weights
    layer = teacher.decoder.layers[0]
    weights = layer.weights[0][:2, :]
    student_layers = student.decoder.layers
    student_layer = student.decoder.layers[0]

    #replace student network with new weights
    test = student_layer.weights[0].numpy()
    test[:2] = weights.numpy()

    #initial new model
    test_model = tf.keras.Sequential()
    test_model.add(tf.keras.layers.InputLayer(input_shape=(8,)))
    l1_initializer = tf.keras.initializers.Constant(test)
    test_model.add(tf.keras.layers.Dense(64, activation='relu', kernel_initializer=l1_initializer))
    for i in range(1, 5):
        test_model.add(student_layers[i])

    vae = Merge_VAE(latent_dim=8, encoder=student.encoder, decoder=test_model)
    return vae

def restore_network(loc):
    model = CVAE(latent_dim=8, beta=6, shape=[28, 28, 1], model='without_bias')
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore("./checkpoints/{}".format(loc))
    return model

def adjust_learn(epochs, merge_network, full_range_set, partial_range_set, date, filePath):
    @tf.function
    def train_step(model, x, degree_set, optimizer):
        s = degree_set[0]
        e = degree_set[1]
        for i in range(s, e + 10, 10):
            d = np.radians(i)
            with tf.GradientTape() as tape:
                r_x = rotate(x, d)
                ori_loss = compute_loss(model, x)
                rota_loss = reconstruction_loss(model, r_x)
                ori_cross_l = ori_cross_loss(model, x, d, r_x)
                rota_cross_l = rota_cross_loss(model, x, d, r_x)
                total_loss = ori_loss + rota_loss + ori_cross_l + rota_cross_l
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    checkpoint_path = "./checkpoints/"+ date + filePath

    ckpt = tf.train.Checkpoint(merge_network=merge_network,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    loss = tf.keras.metrics.Mean()
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    for test_batch in partial_range_set.take(1):
        test_sample = test_batch[0:num_examples_to_generate, :, :, :]
    for epoch in range(epochs):
        for train_p in partial_range_set:
            train_step(merge_network, train_p, [0,180], optimizer)

        for train_x in full_range_set:
            train_step(merge_network, train_x, [180,360], optimizer)

        if (epoch+1)%10 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                    ckpt_save_path))
            for i in range(10, 370, 10):
                d = np.radians(i)
                r_x = rotate(test_sample, d)
                ori_loss = compute_loss(merge_network, test_sample)
                rota_loss = reconstruction_loss(merge_network, test_sample)
                ori_cross_l = ori_cross_loss(merge_network, test_sample, d, r_x)
                rota_cross_l = rota_cross_loss(merge_network, test_sample, d, r_x)
                total_loss = ori_loss + rota_loss + ori_cross_l + rota_cross_l
                loss(total_loss)
            elbo = -loss.result()
            print('Epoch: {}, Test set ELBO: {}'
                  .format(epoch+1, elbo))



if __name__ == '__main__':
    (mnist_images, mnist_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    mnist_images = preprocess_images(mnist_images)
    batch_size = 32
    tmp = np.zeros(shape=[1, 28, 28, 1]).astype('float32')
    tmp[:, :, 13] = 1
    full_range = tmp
    partial_range = mnist_images[np.where(mnist_labels == 3)][:100]
    partial_range_digit = (tf.data.Dataset.from_tensor_slices(partial_range)
                           .batch(batch_size))

    full_range_digit = (tf.data.Dataset.from_tensor_slices(full_range)
                        .batch(batch_size))

    teacher = restore_network("6_9/teacher_network3/ckpt-8")
    student = restore_network("6_9/student_network3/ckpt-8")
    merge_network  = merge_network(teacher, student)
    optimizer = tf.keras.optimizers.Adam(1e-4)
    date = '6_9/'
    file_path = 'merge_network2/'
    epochs = 30
    num_examples_to_generate = 16

    adjust_learn(epochs, merge_network, full_range_digit, partial_range_digit, date, file_path)