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

    #logx_z = cross_entropy(phi_x, r_x)



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




def start_train(epochs, model, train_dataset, test_dataset, date, filePath):
    @tf.function
    def train_step(model, x, optimizer):
        for degree in range(40):
            with tf.GradientTape() as tape:
                d = np.radians((degree + 1)*6)
                r_x = rotate(x, d)
                ori_loss = compute_loss(model, x)
                rota_loss = reconstruction_loss(model, r_x)
                ori_cross_l = ori_cross_loss(model, x, d, r_x)
                rota_cross_l = rota_cross_loss(model, x, d, r_x)
                total_loss = ori_loss + rota_loss + ori_cross_l + rota_cross_l
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        '''
        with tf.GradientTape() as tape:
            r_x = rotate(x, d)
            rota_loss = compute_loss(model, r_x)
        gradients = tape.gradient(rota_loss, model.trainable_variables)  
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        with tf.GradientTape() as tape:
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        '''
    checkpoint_path = "./checkpoints/"+ date + filePath
    ckpt = tf.train.Checkpoint(model=model,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    display.clear_output(wait=False)
    for test_batch in test_dataset.take(1):
        test_sample = test_batch[0:num_examples_to_generate, :, :, :]
    for epoch in range(epochs):
        generate_and_save_images(model, 0, test_sample, file_path)
        start_time = time.time()
        for train_x in train_dataset:
            train_step(model, train_x, optimizer)
        end_time = time.time()
        loss = tf.keras.metrics.Mean()
        #generate_and_save_images(model, epochs, test_sample, file_path)
        #generate_and_save_images(model, epochs, r_sample, "rotate_image")
        # generate_and_save_images(model, epochs, r_sample, "rotate_image")
        if (epoch + 1) % 1 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

            for test_x in test_dataset:
                d = np.radians(random.randint(30, 90))
                r_x = rotate(test_x, d)
                total_loss = rota_cross_loss(model, test_x, d, r_x) \
                             + ori_cross_loss(model, test_x, d, r_x) \
                             + compute_loss(model, test_x) \
                             + reconstruction_loss(model, r_x)
                loss(total_loss)
            elbo = -loss.result()
            print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
                  .format(epochs, elbo, end_time - start_time))

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

    print('Keys in the dataset:', dataset_zip.keys())
    imgs = dataset_zip['imgs']
    imgs = np.reshape(imgs, [len(imgs), 64, 64, 1]).astype('float32')
    latents_values = dataset_zip['latents_values']
    latents_classes = dataset_zip['latents_classes']
    latents_classes = pd.DataFrame(latents_classes)
    latents_classes.columns = ["color", "shape", "scale", "orientation", "x_axis", "y_axis"]
    train_images_index = latents_classes[latents_classes.orientation==0].index
    test_images_index = latents_classes[latents_classes.orientation==39].index
    train_images = imgs[train_images_index]
    test_images = imgs[test_images_index]
    latent_dim = 8
    num_examples_to_generate = 16
    test_size = 10000
    random_vector_for_generation = tf.random.normal(
        shape=[num_examples_to_generate, latent_dim])
    epochs = 30
    model = CVAE(latent_dim=latent_dim, beta=4, shape=[64,64,1])
    sample_size = 1000
    train_size = sample_size * 10
        #train_size = 10000
        #train_images = train_set
    batch_size = 32
    train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                     .shuffle(train_size).batch(batch_size))
    test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                    .shuffle(test_size).batch(batch_size))
    date = '6_13/'
    file_path = 'sprite_size'
    start_train(epochs, model, train_dataset, test_dataset, date, file_path)


