import tensorflow as tf
from model import CVAE, Classifier, SIM_CLR
from dataset import preprocess_images, divide_dataset, imbalance_sample
from tensorflow_addons.image import rotate
import time
from tensorflow.linalg import matvec
import matplotlib.pyplot as plt
import numpy as np
import os
from IPython import display
import math

optimizer = tf.keras.optimizers.Adam(1e-4)
mbs = tf.losses.MeanAbsoluteError()
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)



def reconstruction_loss(model, X, y):
    mean, logvar = model.encode(X)
    Z = model.reparameterize(mean, logvar)
    X_pred = model.decode(Z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=X_pred, labels=X)
    logx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    h = model.projection(Z)
    encode_loss = top_loss(model, h, y)
    return -tf.reduce_mean(logx_z) + encode_loss, h


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def rotate_vector(vector, matrix):
    matrix = tf.cast(matrix, tf.float32)
    test = matvec(matrix, vector)
    return test


def ori_cross_loss(model, x, d, r_x, y):
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

    h = model.projection(phi_z)
    encode_loss = top_loss(model, h, y)


    return -tf.reduce_mean(logx_z) + encode_loss


def rota_cross_loss(model, x, d, r_x, y):
    c, s = np.cos(d), np.sin(d)
    latent = model.latent_dim
    r_m = np.identity(latent)
    r_m[0, [0, 1]], r_m[1, [0, 1]] = [c, s], [-s, c]
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    phi_z = rotate_vector(z, r_m)
    phi_x = model.decode(phi_z)

    h = model.projection(phi_z)
    encode_loss = top_loss(model, h, y)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=phi_x, labels=r_x)
    logx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])

    return -tf.reduce_mean(logx_z) + encode_loss




def compute_loss(model, x, y):
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
    h = model.projection(z)
    encode_loss = top_loss(model, h, y)
    return -tf.reduce_mean(logx_z + beta * (logpz - logqz_x)) + encode_loss, h


def top_loss(model, h, y):
    classes = model.num_cls
    labels = tf.one_hot(y, classes)
    loss_t = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=labels, logits=h
    ))

    return loss_t




def start_train(epochs, model, train_set, majority_set, test_set, date, filePath):
    @tf.function
    def train_step(model, x, y, optimizer, oversample=False):
        if (oversample):
            with tf.GradientTape() as tape:
                mean, logvar = model.encode(x)
                z = model.reparameterize(mean, logvar)
                c, s = np.cos(d), np.sin(d)
                latent = model.latent_dim
                r_m = np.identity(latent)
                r_m[0, [0, 1]], r_m[1, [0, 1]] = [c, s], [-s, c]
                r_z = rotate_vector(z, r_m)
                r_x = model.sample(r_z)
                '''
                r_mean, r_logvar = model.encode(r_x)
                r_x_z = model.reparameterize(r_mean, r_logvar)
                h = model.projection(r_x_z)
                encode_loss = top_loss(model, h, y)
                '''
                ori_loss, _ = compute_loss(model, x, y)
                total_loss = ori_loss
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        else:
            with tf.GradientTape() as tape:
                ori_loss, _ = compute_loss(model, x, y)
                total_loss = ori_loss
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    checkpoint_path = "./checkpoints/"+ date + filePath
    ckpt = tf.train.Checkpoint(sim_clr=model,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    display.clear_output(wait=False)
    for epoch in range(epochs):

        start_time = time.time()

        for x, y in tf.data.Dataset.zip((train_set[0], train_set[1])):
            train_step(model, x, y, optimizer)

        #for x, y in tf.data.Dataset.zip((majority_set[0], majority_set[1])):
        #    train_step(model, x, y, optimizer)


        end_time = time.time()
        elbo_loss = tf.keras.metrics.Mean()
        acc = tf.keras.metrics.Mean()
        #generate_and_save_images(model, epochs, r_sample, "rotate_image")
        if (epoch + 1)%5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                        ckpt_save_path))
            ori_loss, h = compute_loss(model, test_set[0], test_set[1])
            correct_r_h = np.sum(h.numpy().argmax(-1) == test_set[1])
            percentage = (correct_r_h/float(len(test_set[1])))
            total_loss = ori_loss
            elbo_loss(total_loss)
            acc(percentage)
            elbo =  -elbo_loss.result()
            avage_acc = acc.result()
            print('Epoch: {}, elbo: {}, accuracy: {}, time elapse for current epoch: {}'
                  .format(epoch+1, elbo, avage_acc, end_time - start_time))

    #compute_and_save_inception_score(model, file_path)


def divide_array(l):
    m = int(math.floor(len(l))/2)
    head = l[:m]
    tail = l[m:m*2]
    return head, tail

def pre_cast(dataset, digits):
    tmp =  dataset[np.where(np.isin(mnist_labels, [digits[0]]))]
    head, tail  = divide_array(tmp)
    labels = np.array([0]*len(head))
    for i in range(1, len(digits)-1):
        array = mnist_images[np.where(np.isin(mnist_labels, [digits[i]]))]
        h, t = divide_array(array)
        l = np.array((i) * len(h))
        head = np.concatenate([head, h])
        tail = np.concatenate([tail, t])
        labels = np.concatenate([labels, l])
    return head, tail,labels



if __name__ == '__main__':
    (mnist_images, mnist_labels), (test_images, testset_labels) = tf.keras.datasets.mnist.load_data()
    mnist_images = preprocess_images(mnist_images)
    test_images = preprocess_images(test_images)
    irs = [4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40]
    majority_images = mnist_images[np.where(mnist_labels==0)][irs[0]]
    majority_labels = [0] * irs[0]
    train_images, train_labels = imbalance_sample(mnist_images, mnist_labels, irs)
    num_examples_to_generate = 16
    epochs = 50
    batch_size = 32
    sim_clr = SIM_CLR(model='mlp')
    train_images = (tf.data.Dataset.from_tensor_slices(train_images)
            .shuffle(len(train_images), seed=1).batch(batch_size))

    train_labels = (tf.data.Dataset.from_tensor_slices(train_labels)
                    .shuffle(len(train_labels), seed=1).batch(batch_size))

    majority_images = (tf.data.Dataset.from_tensor_slices(majority_images)
            .shuffle(len(majority_images), seed=1).batch(batch_size))




    date = '7_7/'
    file_path = 'mnist_test1/'
    start_train(epochs, sim_clr, [train_images, train_labels], [majority_images, majority_labels],
                [test_images, testset_labels], date, file_path)