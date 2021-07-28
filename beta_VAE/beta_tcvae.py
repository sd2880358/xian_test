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



def kl_divergence(mean, logvar):
    summand = tf.math.square(mean) + tf.math.exp(logvar) - logvar  - 1
    return (0.5 * tf.reduce_sum(summand, [1]))

def compute_loss(model, x):
    beta = model.beta
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logx_z = tf.reduce_mean(tf.reduce_sum(cross_ent, axis=[1, 2, 3]))
    log_qz, logq_z_product = estimate_entropies(z, mean, logvar)
    tc = tf.reduce_mean(log_qz - logq_z_product)
    kl_loss = tf.reduce_mean(kl_divergence(mean, logvar))

    return tf.reduce_mean(logx_z + kl_loss + (beta-1) * tc)

def gaussian_log_density(samples, mean, logvar):
    pi = tf.constant(np.pi)
    normalization = tf.math.log(2. * pi)
    inv_sigma = tf.math.exp(-logvar)
    tmp = (samples - mean)
    return -0.5 * (tmp * tmp * inv_sigma + logvar + normalization)


def estimate_entropies(qz_samples, mean, logvar):
    log_q_z_prob = gaussian_log_density(
        tf.expand_dims(qz_samples,1),  tf.expand_dims(mean,0),
    tf.expand_dims(logvar, 0))

    log_q_z_product = tf.math.reduce_sum(
        tf.math.reduce_logsumexp(log_q_z_prob, axis=1, keepdims=False),
        axis=1, keepdims=False
    )

    log_qz = tf.math.reduce_logsumexp(
        tf.math.reduce_sum(log_q_z_prob, axis=2, keepdims=False)
    )
    return log_qz, log_q_z_product

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


def ori_cross_loss(model, x, w, r_x):
    mean, logvar = model.encode(r_x)
    w_z = model.reparameterize(mean, logvar)
    z = w_z/w
    phi_x = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=phi_x, labels=x)
    logx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])

    return -tf.reduce_mean(logx_z)


def rota_cross_loss(model, x, w, r_x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    w_z = z * w
    phi_x = model.decode(w_z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=phi_x, labels=r_x)
    logx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])

    return -tf.reduce_mean(logx_z)

def reconstruction_loss(model, X):
    mean, logvar = model.encode(X)
    Z = model.reparameterize(mean, logvar)
    X_pred = model.decode(Z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=X_pred, labels=X)
    logx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    return -tf.reduce_mean(logx_z)


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




def start_train(epochs, model, full_range_set, partial_range_set, full, partial, date, filePath):
    def train_step(model, x, degree_set, optimizer, sample_index):
        s = degree_set[0]
        e = degree_set[1]
        for size in range(s+1, e+1):
            sample_set = imgs[sample_index[size]]
            with tf.GradientTape() as tape:
                ori_loss = compute_loss(model, test_sample)
                rota_loss = reconstruction_loss(model, sample_set)
                ori_cross_l = ori_cross_loss(model, x, size, sample_set)
                rota_cross_l = rota_cross_loss(model, x, size, sample_set)
                total_loss = ori_loss + rota_cross_l + ori_cross_l + rota_loss
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
    for epoch in range(epochs):
        start_time = time.time()

        for train_x in full_range_set:
            train_step(model, train_x, [1, 4], optimizer, full)

        for train_p in partial_range_set:
            train_step(model, train_p, [0,1], optimizer, partial)



        end_time = time.time()
        loss = tf.keras.metrics.Mean()

        if (epoch + 1)%100 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                        ckpt_save_path))

            for size in range(1, 5):
                w_sample = imgs[test_set[size]]
                ori_loss = compute_loss(model, test_sample)
                rota_loss = reconstruction_loss(model, test_sample)
                ori_cross_l = ori_cross_loss(model, test_sample, size, w_sample)
                rota_cross_l = rota_cross_loss(model, test_sample, size, w_sample)
                total_loss = ori_loss + rota_loss + ori_cross_l + rota_cross_l
                loss(total_loss)

            elbo = -loss.result()
            #generate_and_save_images(model, epoch, test_sample, file_path)
            print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
                  .format(epoch+1, elbo, end_time - start_time))


    #compute_and_save_inception_score(model, file_path)





def calculate_fid(real, fake):
    mu1, sigma1 = real.mean(axis=0), np.cov(real, rowvar=False)
    mu2, sigma2 = fake.mean(axis=0), np.cov(fake, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid




if __name__ == '__main__':
    dataset_zip = np.load('../dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    imgs = dataset_zip['imgs']
    imgs = np.reshape(imgs, [len(imgs), 64, 64, 1]).astype('float32')
    latents_values = dataset_zip['latents_values']
    latents_classes = dataset_zip['latents_classes']
    latents_classes = pd.DataFrame(latents_classes)
    latents_classes.columns = ["color", "shape", "scale", "orientation", "x_axis", "y_axis"]
    full = []
    partial = []
    test_set = []
    for i in range(5):
        full_index = latents_classes.loc[((latents_classes['shape'] == 2) &
                                          (latents_classes['scale'] == i) &
                                          (latents_classes['x_axis'] == 15) &
                                          (latents_classes['y_axis'] == 15))].index
        full.append(full_index)
        partial_index = latents_classes.loc[((latents_classes['shape'] == 0) &
                                                 (latents_classes['scale'] == i) &
                                                 (latents_classes['x_axis'] == 15) &
                                                 (latents_classes['y_axis'] == 15))].index
        test_set.append(partial_index)
        if (i < 2):
                partial.append(partial_index.append(full_index))

    full = pd.DataFrame(full).T.sample(frac=1)
    partial = pd.DataFrame(partial).T.sample(frac=1)
    full_images = imgs[full[0]]
    partial_images = imgs[partial[0]]
    test_sample = imgs[test_set[0]]
    latent_dim = 8
    num_examples_to_generate = 16
    random_vector_for_generation = tf.random.normal(
        shape=[num_examples_to_generate, latent_dim])
    epochs = 600
    model = CVAE(latent_dim=latent_dim, beta=4, shape=[64, 64, 1])
    batch_size = 40
    full_dataset = (tf.data.Dataset.from_tensor_slices(full_images)
                     .batch(batch_size))
    partial_dataset = (tf.data.Dataset.from_tensor_slices(partial_images)
                    .batch(80))

    date = '6_14/'
    file_path = 'sprite_size5'
    start_train(epochs, model, full_dataset, partial_dataset, full, partial, date, file_path)


