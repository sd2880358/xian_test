from model import Communication, F_VAE
import tensorflow as tf
from tensorflow_addons.image import rotate
import time
import matplotlib.pyplot as plt
import numpy as np
import os
from IPython import display
import math
import pandas as pd
from loss import compute_loss





def top_loss(model, x, y):
    num_cls = model.num_cls
    h = model.projection(x)
    tmp = 0
    for i in range(len(num_cls)):
        labels = tf.one_hot(y[:, i], num_cls[i])
        loss_t = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=labels, logits=h[i]
        ), axis=0)
        tmp = tf.math.add(loss_t, tmp)
    return tmp


def start_train(epochs, model, train_set, test_set, date, filePath, count):
    @tf.function
    def train_step(model, x, y, optimizer):
        with tf.GradientTape() as tape:
            ori_loss = top_loss(model, x, y)
            total_loss = ori_loss
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    checkpoint_path = "./checkpoints/{}/{}/test{}".format(date, filePath, count)
    ckpt = tf.train.Checkpoint(classifier=model,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    display.clear_output(wait=False)
    e = 0
    file_dir = "./score/{}/{}".format(date, filePath)
    file = file_dir + '{}.csv'.format(count)
    if os.path.isfile(file):
        e = pd.read_csv(file).index[-1]
    for epoch in range(epochs):

        start_time = time.time()

        for x, y in tf.data.Dataset.zip((train_set[0], train_set[1])):
            train_step(model, x, y, optimizer)

        #for x, y in tf.data.Dataset.zip((majority_set[0], majority_set[1])):
        #    train_step(model, x, y, optimizer)


        end_time = time.time()
        d_type_acc = tf.keras.metrics.Mean()
        app_acc = tf.keras.metrics.Mean()
        mob_acc = tf.keras.metrics.Mean()
        acc = [d_type_acc, app_acc, mob_acc]
        e += 1
        #generate_and_save_images(model, epochs, r_sample, "rotate_image")
        if (epoch +1)%10 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                        ckpt_save_path))
            h = model.projection(test_set[0])
            for i in range(3):
                correct_r_h = np.sum(h[i].numpy()[i].argmax(-1) == test_set[1][:, i])
                percentage = (correct_r_h/float(len(test_set[1][:, i])))
                acc[i](percentage)
            accuracy = [i.result().numpy() for i in acc]
            average = sum(accuracy)/3
            print('Epoch: {},  average accuracy: {}, time elapse for current epoch: {}'
                  .format(epoch+1, average, end_time - start_time))
            df = pd.DataFrame({
                "device_type": accuracy[0],
                "app": accuracy[1],
                'mob': accuracy[2]
            }, index=[e], dtype=np.float32)
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            if not os.path.isfile(file):
                df.to_csv(file)
            else:  # else it exists so append without writing the header
                df.to_csv(file, mode='a', header=False)


    #compute_and_save_inception_score(model, file_path)


if __name__ == '__main__':
    date = '7_12'
    dims = [120, 168]
    batch_size = 6
    epochs = 50
    optimizer = tf.keras.optimizers.Adam(1e-4)
    for data_index in range(2):
        file = np.load('../communication_data/dataset{}.npz'.format(data_index))
        dataset = file['dataset']
        labelset = file['labelset']
        classifier = Communication(shape=[dims[data_index], 6, 1])

        for j in range(5):
            file_path = ('mix_cnn_test{}').format(data_index)
            seed = np.random.randint(0, 100)
            train_size = math.ceil(len(dataset) * 0.8)
            train_set, train_labels = dataset[:train_size, :, :, :], labelset[:train_size, :]
            test_set, test_labels = dataset[train_size:, :, :, :], labelset[train_size:, :]
            train_set = (tf.data.Dataset.from_tensor_slices(train_set)
                            .batch(batch_size))
            train_labels = (tf.data.Dataset.from_tensor_slices(train_labels)
                        .batch(batch_size))
            start_train(epochs, classifier, [train_set, train_labels],
                        [test_set, test_labels], date, file_path, j)