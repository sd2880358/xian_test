import tensorflow as tf
from model import CVAE, Classifier, F_VAE
from dataset import preprocess_images, divide_dataset, imbalance_sample
from tensorflow_addons.image import rotate
import time
from tensorflow.linalg import matvec
import matplotlib.pyplot as plt
import numpy as np
import os
from IPython import display
import math

sim_optimizer = tf.keras.optimizers.Adam(1e-4)
cls_optimizer = tf.keras.optimizers.Adam(1e-4)
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


def kl_divergence(mean, logvar):
    summand = tf.math.square(mean) + tf.math.exp(logvar) - logvar  - 1
    return (0.5 * tf.reduce_sum(summand, [1]))

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


def compute_loss(model, classifier, x, y):
    beta = model.beta
    mean, logvar = model.encode(x)
    features = model.reparameterize(mean, logvar)
    identity = tf.expand_dims(tf.cast(y, tf.float32), 1)
    z = tf.concat([features, identity], axis=1)
    x_logit = model.decode(z)
    '''
    reco_loss = reconstruction_loss(x_logit, x)
    kl_loss = kl_divergence(logvar, mean)
    beta_loss = reco_loss + kl_loss * beta
    '''
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logx_z = tf.reduce_mean(tf.reduce_sum(cross_ent, axis=[1, 2, 3]))
    log_qz, logq_z_product = estimate_entropies(features, mean, logvar)

    tc = tf.reduce_mean(log_qz - logq_z_product)
    kl_loss = tf.reduce_mean(kl_divergence(mean, logvar))
    h = classifier.projection(x)
    classifier_loss = top_loss(classifier, h, y)
    x_h = classifier.projection(x_logit)
    gen_loss = top_loss(classifier, x_h, y)
    return tf.reduce_mean(logx_z + kl_loss  + gen_loss + (beta - 1) * tc), h, classifier_loss


def top_loss(model, h, y):
    classes = model.num_cls
    labels = tf.one_hot(y, classes)
    loss_t = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=labels, logits=h
    ))

    return loss_t




def start_train(epochs, model, classifier, train_set, majority_set, test_set, date, filePath):
    @tf.function
    def train_step(model, classifier, x, y, sim_optimizer, cls_optimizer, oversample=False):
        if (oversample):
            with tf.GradientTape() as tape:
                mean, logvar = model.encode(x)
                features = model.reparameterize(mean, logvar)
                num = len(y)
                for id in range(1, 10):
                    ids = [id] * num
                    identity = tf.expand_dims(tf.cast(ids, tf.float32), 1)
                    z = tf.concat([features, identity], axis=1)
                    x_logit = model.decode(z)
                    h = classifier.projection(x_logit)
                    encode_loss = top_loss(classifier, h, ids)
                cls_gradients = tape.gradient(encode_loss, classifier.trainable_variables)
                cls_optimizer.apply_gradients(zip(cls_gradients, classifier.trainable_variables))
        else:
            with tf.GradientTape() as sim_tape, tf.GradientTape() as cls_tape:
                ori_loss, _, encode_loss = compute_loss(model, classifier, x, y)
                total_loss = ori_loss
            sim_gradients = sim_tape.gradient(total_loss, model.trainable_variables)
            cls_gradients = cls_tape.gradient(encode_loss, classifier.trainable_variables)
            cls_optimizer.apply_gradients(zip(cls_gradients, classifier.trainable_variables))
            sim_optimizer.apply_gradients(zip(sim_gradients, model.trainable_variables))
    checkpoint_path = "./checkpoints/"+ date + filePath
    ckpt = tf.train.Checkpoint(sim_clr=model,
                               clssifier = classifier,
                               optimizer=sim_optimizer,
                               cls_optimizer=cls_optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    display.clear_output(wait=False)
    for epoch in range(epochs):

        start_time = time.time()

        for x, y in tf.data.Dataset.zip((train_set[0], train_set[1])):
            train_step(model, classifier, x, y, sim_optimizer, cls_optimizer)

        '''
        for x, y in tf.data.Dataset.zip((majority_set[0], majority_set[1])):
            train_step(model, classifier, x, y, sim_optimizer, cls_optimizer, oversample=True)
        '''


        #for x, y in tf.data.Dataset.zip((majority_set[0], majority_set[1])):
        #    train_step(model, x, y, optimizer)


        end_time = time.time()
        elbo_loss = tf.keras.metrics.Mean()
        acc = tf.keras.metrics.Mean()
        #generate_and_save_images(model, epochs, r_sample, "rotate_image")
        if (epoch +1)%5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                        ckpt_save_path))
            for t_x, t_y in tf.data.Dataset.zip((test_set[0], test_set[1])):
                ori_loss, h, _ = compute_loss(model, classifier, t_x, t_y)
                correct_r_h = np.sum(h.numpy().argmax(-1) == t_y)
                percentage = (correct_r_h/float(len(t_y)))
                total_loss = ori_loss
                elbo_loss(total_loss)
                acc(percentage)
            elbo =  -elbo_loss.result()
            avage_acc = acc.result()
            print('Epoch: {}, elbo: {}, accuracy: {}, time elapse for current epoch: {}'
                  .format(epoch+1, elbo, avage_acc, end_time - start_time))

    #compute_and_save_inception_score(model, file_path)



if __name__ == '__main__':
    (mnist_images, mnist_labels), (test_images, testset_labels) = tf.keras.datasets.mnist.load_data()
    mnist_images = preprocess_images(mnist_images)
    test_images = preprocess_images(test_images)
    irs = [4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40]
    majority_images = mnist_images[np.where(mnist_labels==0)][:irs[0]]
    majority_labels = [0] * irs[0]
    train_images, train_labels = imbalance_sample(mnist_images, mnist_labels, irs)
    num_examples_to_generate = 16
    epochs = 200
    batch_size = 32
    sim_clr = F_VAE(model='cnn')
    classifier = Classifier(shape=[28, 28, 1], model='cnn')

    train_images = (tf.data.Dataset.from_tensor_slices(train_images)
            .shuffle(len(train_images), seed=1).batch(batch_size))

    train_labels = (tf.data.Dataset.from_tensor_slices(train_labels)
                    .shuffle(len(train_labels), seed=1).batch(batch_size))

    majority_images = (tf.data.Dataset.from_tensor_slices(majority_images)
            .shuffle(len(majority_images), seed=2).batch(batch_size))

    majority_labels = (tf.data.Dataset.from_tensor_slices(majority_labels)
            .shuffle(len(majority_labels), seed=2).batch(batch_size))

    test_images = (tf.data.Dataset.from_tensor_slices(test_images)
                    .shuffle(len(test_images), seed=1).batch(batch_size))

    testset_labels = (tf.data.Dataset.from_tensor_slices(testset_labels)
                    .shuffle(len(testset_labels), seed=1).batch(batch_size))


    date = '7_15/'
    file_path = 'mnist_test9/'
    start_train(epochs, sim_clr, classifier, [train_images, train_labels], [majority_images, majority_labels],
                [test_images, testset_labels], date, file_path)
