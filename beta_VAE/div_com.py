from model import Div_Com, C_VAE
import tensorflow as tf
from tensorflow_addons.image import rotate
import time
import matplotlib.pyplot as plt
import numpy as np
import os
from IPython import display
import math
import pandas as pd
from loss import dual_loss, confidence_function, top_loss





def normalization(data):
    data_max = np.max(np.max(data, axis=0), axis=0)
    data_min = np.min(np.min(data, axis=0), axis=0)
    return np.divide(np.add(test_set, np.abs(data_min)), data_max + np.abs(data_min))


def start_train(epochs, classifier, generator, rotation_optimizer, position_optimizer,
                train_set, majority, test_set, date, filePath, count):
    cls_optimizer = tf.keras.optimizers.Adam(1e-4)
    def train_step(classifier, generator, x, y, cls_optimizer, rotation_optimizer,
                   position_optimizer, oversample=False, threshold=None):
        if (oversample):
            with tf.GradientTape() as rotation_tape, tf.GradientTape() as position_tape, tf.GradientTape() as cls_tape:
                mean, logvar = generator.encode(x)
                features = generator.reparameterize(mean, logvar)
                num = len(y)
                for id in range(1, 10):
                    ids = [id] * num
                    identity = tf.expand_dims(tf.cast(ids, tf.float32), 1)
                    z = tf.concat([features, identity], axis=1)
                    x_logit = generator.sample(z)
                    conf, l = confidence_function(classifier, x_logit)
                    x_labels = classifier.projection(x_logit).numpy()
                    mis_sample = x_labels[np.where(l!=id)]
                    mis_sample_loss = top_loss(classifier, h=mis_sample, y=[id]*len(mis_sample))
                    sample = x_logit.numpy()[np.where(conf>=threshold)]
                    l = l[np.where(conf>=threshold)]
                    rotation_loss, position_loss, h, cls_loss = dual_loss(generator, classifier, sample, l, gamma=1)

                rotation_gradients = rotation_tape.gradient(rotation_loss+mis_sample_loss,
                                                            generator.encoder.trainable_variables +
                                                            generator.rotation.trainable_variables)
                position_gradients = position_tape.gradient(position_loss+mis_sample_loss,
                                                            generator.encoder.trainable_variables +
                                                            generator.position.trainable_variables)
                cls_gradients = cls_tape.gradient(cls_loss, classifier.trainable_variables)
                cls_optimizer.apply_gradients(zip(cls_gradients, classifier.trainable_variables))
                rotation_optimizer.apply_gradients(zip(rotation_gradients, generator.encoder.trainable_variables +
                                                       generator.rotation.trainable_variables))
                position_optimizer.apply_gradients(zip(position_gradients, generator.encoder.trainable_variables +
                                                       generator.position.trainable_variables))
        else:
            with tf.GradientTape() as rotation_tape, tf.GradientTape() as position_tape, tf.GradientTape() as cls_tape:
                rotation_loss, position_loss, h, cls_loss = dual_loss(generator, classifier, x, y, gamma=1)
            rotation_gradients = rotation_tape.gradient(rotation_loss, generator.encoder.trainable_variables +
                                                   generator.rotation.trainable_variables)
            position_gradients = position_tape.gradient(position_loss, generator.encoder.trainable_variables +
                                                   generator.position.trainable_variables)
            cls_gradients = cls_tape.gradient(cls_loss, classifier.trainable_variables)
            cls_optimizer.apply_gradients(zip(cls_gradients, classifier.trainable_variables))
            rotation_optimizer.apply_gradients(zip(rotation_gradients, generator.encoder.trainable_variables +
                                                   generator.rotation.trainable_variables))
            position_optimizer.apply_gradients(zip(position_gradients, generator.encoder.trainable_variables +
                                                   generator.position.trainable_variables))
    checkpoint_path = "./checkpoints/{}/{}/test{}".format(date, filePath, count)
    ckpt = tf.train.Checkpoint(classifier=classifier, generator=generator, cls_optimizer=cls_optimizer,
                               rotation_optimizer=rotation_optimizer,
                               position_optimizer=position_optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    display.clear_output(wait=False)
    e = 0
    file_dir = "./score/{}/{}".format(date, filePath)
    file = file_dir + '/test{}.csv'.format(count)
    if os.path.isfile(file):
        e = pd.read_csv(file).index[-1]
    for epoch in range(epochs):
        start_time = time.time()
        for x, y in tf.data.Dataset.zip((train_set[0], train_set[1])):
            train_step(classifier, generator, x, y, cls_optimizer, rotation_optimizer, position_optimizer)

        '''
        for x, y in tf.data.Dataset.zip((majority[0], majority[1])):
            train_step(classifier, generator, x, y, cls_optimizer,
                       rotation_optimizer, position_optimizer, oversample=True, threshold=0.75)
        '''
        end_time = time.time()
        rot_loss = tf.keras.metrics.Mean()
        pos_loss = tf.keras.metrics.Mean()
        e += 1
        labels_acc = []
        #generate_and_save_images(model, epochs, r_sample, "rotate_image")
        if (epoch +1)%10 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                        ckpt_save_path))
            rotation_loss, position_loss, h, cls_loss = dual_loss(generator, classifier, test_set[0], test_set[1], gamma=1)
            correct_r_h = np.sum(h.numpy().argmax(-1) == test_set[1])
            for label in range(classifier.num_cls):
                correct_label = np.sum((h.numpy().argmax(-1) == test_set[1])[np.where(test_set[1]==label)])
                if (correct_label == 0):
                    labels_acc.append(0)
                else:
                    labels_acc.append(correct_label/float(np.sum(test_set[1]==label)))
            percentage = (correct_r_h/float(len(test_set[1])))
            rot_loss(rotation_loss)
            pos_loss(position_loss)
            ro_loss = -rot_loss.result()
            po_loss = -pos_loss.result()
            acc_array = {}
            acc_array['total_acc'] = percentage
            if (acc_array['total_acc'] >= 0.9):
                cls_optimizer = tf.keras.optimizers.Adam(1e-5)
            for i in range(len(labels_acc)):
                acc_array['label{}'.format(i)] = labels_acc[i]
            print('Epoch: {},  rotation_loss: {}, position_loss:{}, average accuracy: {}, time elapse for current epoch: {}'
                  .format(epoch+1, ro_loss, po_loss, acc_array['total_acc'], end_time - start_time))
            df = pd.DataFrame(
                acc_array
            , index=[e], dtype=np.float32)
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            if not os.path.isfile(file):
                df.to_csv(file)
            else:  # else it exists so append without writing the header
                df.to_csv(file, mode='a', header=False)


    #compute_and_save_inception_score(model, file_path)


if __name__ == '__main__':
    rotation_optimizer = tf.keras.optimizers.Adam(1e-4)
    position_optimizer = tf.keras.optimizers.Adam(1e-4)
    batch_size = 6
    epochs = 100
    label_index = ['device_type', 'application', 'mobility']
    num_class_index = [3, 4, 3]
    date = '7_27'
    test_id = 'oversample_test1'
    dims = [120, 168, 600, 240]
    for data_index in range(4):
        file = np.load('../communication_data/dataset{}.npz'.format(data_index))
        dataset = file['dataset']
        labelset = file['labelset']
        for i in range(3):
            train_size = math.ceil(len(dataset) * 0.8)
            train_set, train_labels = dataset[:train_size, :, :, :], labelset[:train_size, i]
            test_set, test_labels = dataset[train_size:, :, :, :], labelset[train_size:, i]
            majority_label = np.bincount(train_labels.flatten()).argmax(-1)
            majority_dataset = train_set[np.where(train_labels == majority_label)]
            majority_labelset = [majority_label] * len(majority_dataset)
            train_set = (tf.data.Dataset.from_tensor_slices(train_set)
                         .batch(batch_size))
            train_labels = (tf.data.Dataset.from_tensor_slices(train_labels)
                            .batch(batch_size))
            majority_set = (tf.data.Dataset.from_tensor_slices(majority_dataset)
                         .batch(batch_size))
            majority_labels = (tf.data.Dataset.from_tensor_slices(majority_labelset)
                            .batch(batch_size))
            for j in range(5):
                classifier = Div_Com(shape=[dims[data_index], 6, 1], num_cls=num_class_index[i], model='cnn')
                generator = C_VAE(shape=[dims[data_index], 6, 1], latent_dim=8, num_cls=num_class_index[i])
                file_path = (test_id+'{}/{}').format(data_index, label_index[i])

                start_train(epochs, classifier, generator,rotation_optimizer, position_optimizer,
                            [train_set, train_labels], [majority_set, majority_labels],
                            [test_set, test_labels], date, file_path, j)