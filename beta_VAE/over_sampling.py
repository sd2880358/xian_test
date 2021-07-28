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
import pandas as pd
from loss import compute_loss, confidence_function, top_loss, acc_metrix

def start_train(epochs, target, threshold, model, classifier, train_set, majority_set, test_set, date, filePath):
    sim_optimizer = tf.keras.optimizers.Adam(1e-4)
    cls_optimizer = tf.keras.optimizers.Adam(1e-4)
    def train_step(model, classifier, x, y, sim_optimizer, cls_optimizer, oversample=False, threshold=None):
        if (oversample):
            with tf.GradientTape() as sim_tape, tf.GradientTape() as cls_tape:
                mean, logvar = model.encode(x)
                features = model.reparameterize(mean, logvar)
                num = len(y)
                for id in range(1, 10):
                    ids = [id] * num
                    identity = tf.expand_dims(tf.cast(ids, tf.float32), 1)
                    z = tf.concat([features, identity], axis=1)
                    x_logit = model.sample(z)
                    conf, l = confidence_function(classifier, x_logit, target=target)
                    x_labels = classifier.projection(x_logit).numpy()
                    mis_sample = x_labels[np.where((l!=id) & (conf>=threshold))]
                    mis_sample_loss = top_loss(classifier, h=mis_sample, y=[id]*len(mis_sample))
                    sample = x_logit.numpy()[np.where((conf>=threshold) & (l==id))]
                    ori_loss, h, cls_loss = compute_loss(model, classifier, sample, [id]*len(sample), gamma=1)
                sim_gradients = sim_tape.gradient(ori_loss, model.trainable_variables)
                sim_optimizer.apply_gradients(zip(sim_gradients, model.trainable_variables))
                '''
                cls_gradients = cls_tape.gradient(cls_loss, classifier.trainable_variables)
                cls_optimizer.apply_gradients(zip(cls_gradients, classifier.trainable_variables))
                '''
        else:
            with tf.GradientTape() as sim_tape, tf.GradientTape() as cls_tape:
                ori_loss, _, encode_loss = compute_loss(model, classifier, x, y)
                total_loss = ori_loss
            sim_gradients = sim_tape.gradient(total_loss, model.trainable_variables)
            cls_gradients = cls_tape.gradient(encode_loss, classifier.trainable_variables)
            cls_optimizer.apply_gradients(zip(cls_gradients, classifier.trainable_variables))
            sim_optimizer.apply_gradients(zip(sim_gradients, model.trainable_variables))
    checkpoint_path = "./checkpoints/{}/{}".format(date, filePath)
    ckpt = tf.train.Checkpoint(sim_clr=model,
                               clssifier = classifier,
                               optimizer=sim_optimizer,
                               cls_optimizer=cls_optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    display.clear_output(wait=False)

    result_dir = "./score/{}/{}".format(date, filePath)
    if os.path.isfile(result_dir+'/result.csv'):
        e = pd.read_csv(result_dir+'/result.csv').index[-1]
    else:
        e = 0
    for epoch in range(epochs):

        e += 1
        start_time = time.time()

        for x, y in tf.data.Dataset.zip((train_set[0], train_set[1])):
            train_step(model, classifier, x, y, sim_optimizer, cls_optimizer)


        for x, y in tf.data.Dataset.zip((majority_set[0], majority_set[1])):
            train_step(model, classifier, x, y, sim_optimizer, cls_optimizer, oversample=True, threshold=threshold)



        #for x, y in tf.data.Dataset.zip((majority_set[0], majority_set[1])):
        #    train_step(model, x, y, optimizer)


        end_time = time.time()
        elbo_loss = tf.keras.metrics.Mean()
        over_sample_loss = tf.keras.metrics.Mean()
        #generate_and_save_images(model, epochs, r_sample, "rotate_image")
        if (epoch +1)%5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                        ckpt_save_path))
            for t_x, t_y in tf.data.Dataset.zip((test_set[0], test_set[1])):
                ori_loss, h, _ = compute_loss(model, classifier, t_x, t_y)
                g_mean, acsa = acc_metrix(h.numpy().argmax(-1), t_y.numpy())
                total_loss = ori_loss
                elbo_loss(total_loss)

            for x, y in tf.data.Dataset.zip((majority_set[0], majority_set[1])):
                mean, logvar = model.encode(x)
                features = model.reparameterize(mean, logvar)
                num = len(y)
                for id in range(1, 10):
                    ids = [id] * num
                    identity = tf.expand_dims(tf.cast(ids, tf.float32), 1)
                    z = tf.concat([features, identity], axis=1)
                    x_logit = model.sample(z)
                    conf, l = confidence_function(classifier, x_logit, target=target)
                    sample = x_logit.numpy()[np.where((conf >= threshold) & (l == id))]
                    over_sample_loss(len(sample)/num)
            elbo =  -elbo_loss.result()
            over_sample = over_sample_loss.result()
            df = pd.DataFrame({
                "elbo": elbo,
                "g_mean": g_mean,
                'acsa': acsa,
                'ood': over_sample,
            }, index=[e], dtype=np.float32)
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            if not os.path.isfile(result_dir+'/result.csv'):
                df.to_csv(result_dir+'/result.csv')
            else:  # else it exists so append without writing the header
                df.to_csv(result_dir+'/result.csv', mode='a', header=False)
            print('Epoch: {}, elbo: {}, g_means: {}, acsa: {}, over_sample_loss: {}, time elapse for current epoch: {}'
                  .format(epoch+1, elbo, g_mean, acsa, over_sample, end_time - start_time))

    #compute_and_save_inception_score(model, file_path)



if __name__ == '__main__':
    target = 'margin'
    threshold = 0.85
    shape = [28, 28, 1]
    mbs = tf.losses.MeanAbsoluteError()
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    (mnist_images, mnist_labels), (test_images, testset_labels) = tf.keras.datasets.mnist.load_data()
    mnist_images = preprocess_images(mnist_images, shape=shape)
    test_images = preprocess_images(test_images, shape=shape)
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


    date = '7_23'
    file_path = 'mnist_test21'
    start_train(epochs, target, threshold, sim_clr, classifier, [train_images, train_labels], [majority_images, majority_labels],
                [test_images, testset_labels], date, file_path)