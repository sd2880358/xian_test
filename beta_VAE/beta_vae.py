import tensorflow as tf
from model import CVAE, Classifier
from dataset import preprocess_images, divide_dataset
from tensorflow_addons.image import rotate
import random
import time
from tensorflow.linalg import matvec
import matplotlib.pyplot as plt
import numpy as np
import os
from IPython import display
import pandas as pd
from scipy.linalg import sqrtm

optimizer = tf.keras.optimizers.Adam(1e-4)
mbs = tf.losses.MeanAbsoluteError()
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def reconstruction_loss(model, X):
    mean, logvar = model.encode(X)
    Z = model.reparameterize(mean, logvar)
    X_pred = model.decode(Z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=X_pred, labels=X)
    logx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    return -tf.reduce_mean(logx_z)


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def rotate_vector(vector, matrix):
    matrix = tf.cast(matrix, tf.float32)
    test = matvec(matrix, vector)
    return test


def ori_cross_loss(model, x, d, r_x):
    mean, logvar = model.encode(r_x)
    r_z = model.reparameterize(mean, logvar)
    c, s = np.cos(d), np.sin(d)
    latent = model.latent_dim
    r_m = np.identity(latent)
    r_m[0, [0, 1]], r_m[1, [0, 1]] = [c, -s], [s, c]
    phi_z = rotate_vector(r_z, r_m)
    phi_x = model.decode(phi_z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=phi_x, labels=x)
    logx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])

    return -tf.reduce_mean(logx_z)


def rota_cross_loss(model, x, d, r_x):
    c, s = np.cos(d), np.sin(d)
    latent = model.latent_dim
    r_m = np.identity(latent)
    r_m[0, [0, 1]], r_m[1, [0, 1]] = [c, s], [-s, c]
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    phi_z = rotate_vector(z, r_m)
    phi_x = model.decode(phi_z)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=phi_x, labels=r_x)
    logx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])

    return -tf.reduce_mean(logx_z)




def compute_loss(model, x):
    beta = model.beta
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    '''
    reco_loss = reconstruction_loss(x_logit, x)
    kl_loss = kl_divergence(logvar, mean)
    beta_loss = reco_loss + kl_loss * beta
    '''
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logx_z + beta * (logpz - logqz_x))


def generate_and_save_images(model, epoch, test_sample, file_path):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')
    file_dir = './image/' + date + file_path
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    plt.savefig(file_dir +'/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()




def start_train(epochs, model, full_range_set, partial_range_set, date, filePath):
    @tf.function
    def train_step(model, x, degree_set, optimizer):
        s = degree_set[0]
        e = degree_set[1]
        for i in range(s, e+10, 10):
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
    ckpt = tf.train.Checkpoint(model=model,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    display.clear_output(wait=False)
    for test_batch in partial_range_set.take(1):
        test_sample = test_batch[0:num_examples_to_generate, :, :, :]
    generate_and_save_images(model, 0, test_sample, file_path)
    for epoch in range(epochs):
        start_time = time.time()

        for train_p in full_range_set:
            train_step(model, train_p, [190, 360], optimizer)


        for train_x in partial_range_set:
            train_step(model, train_x, [0, 180], optimizer)

        end_time = time.time()
        loss = tf.keras.metrics.Mean()


        #generate_and_save_images(model, epochs, r_sample, "rotate_image")
        if (epoch + 1)%10 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                        ckpt_save_path))
            generate_and_save_images(model, epochs, test_sample, file_path)
            for i in range(10, 370, 10):
                d = np.radians(i)
                r_x = rotate(test_sample, d)
                ori_loss = compute_loss(model, test_sample)
                rota_loss = reconstruction_loss(model, test_sample)
                ori_cross_l = ori_cross_loss(model, test_sample, d, r_x)
                rota_cross_l = rota_cross_loss(model, test_sample, d, r_x)
                total_loss = ori_loss + rota_loss + ori_cross_l + rota_cross_l
                loss(total_loss)
            elbo = -loss.result()
            print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
                  .format(epoch+1, elbo, end_time - start_time))

    #compute_and_save_inception_score(model, file_path)






if __name__ == '__main__':
    (mnist_images, mnist_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    mnist_images = preprocess_images(mnist_images)

    partial_range = mnist_images[np.where(np.isin(mnist_labels, [3]))]
    #partial_range = mnist_images[np.where(np.isin(mnist_labels, [4]))]
    tmp = np.zeros(shape=[1000, 28, 28, 1]).astype('float32')
    tmp[:, :, 13] = 1
    full_range = tmp
    num_examples_to_generate = 16
    model = CVAE(latent_dim=8, beta=6, shape=[28, 28, 1])
    epochs = 80
    partial_range = np.concatenate([partial_range, full_range])
    batch_size = 32

    partial_range_digit = (tf.data.Dataset.from_tensor_slices(partial_range)
                         .shuffle(len(full_range)).batch(batch_size))
    full_range_digit = (tf.data.Dataset.from_tensor_slices(full_range)
                         .batch(batch_size))

    date = '6_12/'
    file_path = 'mnist_test21/'
    start_train(epochs, model, full_range_digit, partial_range_digit, date, file_path)