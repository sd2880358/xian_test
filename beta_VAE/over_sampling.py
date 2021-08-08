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
from loss import compute_loss, confidence_function, top_loss, acc_metrix, indices


def estimate(classifier, x_logit, threshold, label, target):
    conf, l = confidence_function(classifier, x_logit, target=target)
    return np.where((conf.numpy()>=threshold) & (l==label))

def merge_list(l1, l2):
    in_l1 = set(l1)
    in_l2 = set(l2)
    in_l1_not_in_l2 = in_l1 - in_l2
    return list(l2) + list(in_l1_not_in_l2)


def latent_triversal(model, classifier, x, y, r, n):
    mean, logvar = model.encode(x)
    features = model.reparameterize(mean, logvar).numpy()
    triversal_range = np.linspace(-r, r, n)
    acc = tf.keras.metrics.Mean()
    for dim in range(features.shape[1]):
        for replace in triversal_range:
            features[:, dim] = replace
            z = tf.concat([features, tf.expand_dims(y,1)], axis=1)
            x_logit = model.sample(z)
            conf, l = confidence_function(classifier, x_logit, target=target)
            sample = x_logit.numpy()[np.where((conf >= threshold) & (l == y))]
            if (len(sample)==0):
                acc(0)
            else:
                acc(len(sample)/len(y))
    return acc.result()

def start_train(epochs, target, threshold_list, method, model, classifier, dataset,
                train_set, test_set, date, filePath):
    optimizer = tf.keras.optimizers.Adam(1e-4)
    checkpoints_list = []
    classifier_list= []
    result_dir_list = []
    metrix_list = []
    if (model.data == 'mnist'):
        file = np.load('../dataset/mnist_oversample_latent.npz')
        latent = file['latent']
        latent_len = latent.shape[0]
        mnist_train_len = np.load('../dataset/mnist_dataset.npz')['train_images'].shape[0]
        block = np.ceil(mnist_train_len/32)
        batch_size = int(np.ceil(latent_len/block))
        latent = (tf.data.Dataset.from_tensor_slices(latent)
                    .shuffle(latent, seed=1).batch(batch_size))
    for i in threshold_list:
        metrix = {}
        metrix['valid_sample'] = []
        metrix['total_sample'] = []
        metrix['total_valid_sample'] = []
        metrix_list.append(metrix)
        result_dir = "./score/{}/{}/{}".format(date, filePath, i)
        result_dir_list.append(result_dir)
        if os.path.isfile(result_dir + '/result.csv'):
            e = pd.read_csv(result_dir + '/result.csv').index[-1]
        checkpoint_path = "./checkpoints/{}/{}/{}".format(date, filePath, i)
        o_classifier = Classifier(shape=dataset.shape, model='mlp', num_cls=dataset.num_cls, threshold=i)
        ckpt = tf.train.Checkpoint(sim_clr=model,
                                   clssifier=classifier,
                                   o_classifier=o_classifier,
                                   )
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')
        classifier_list.append(o_classifier)
        checkpoints_list.append(ckpt_manager)
    def train_step(model, classifier, classifier_list, x,  y, oversample=False, metrix_list=None, features=None):
        if (oversample):
            if(model.data=='celebA' or "large_celebA"):
                mean, logvar = model.encode(x)
                features = model.reparameterize(mean, logvar)
                for cls in range(model.num_cls):
                    # oversampling
                    sample_label = np.array(([cls] * features.shape[0]))
                    z = tf.concat([features, np.expand_dims(sample_label, 1)], axis=1)
                    x_logit = model.sample(z)
                    for i in range(len(classifier_list)):
                        threshold = classifier_list[i].threshold
                        m_index = estimate(classifier, x_logit, threshold, sample_label, target)
                        sample_y = sample_label[m_index]
                        s_index = estimate(classifier_list[i], x_logit, threshold, sample_label, target)
                        o_sample_y = sample_label[s_index]
                        #total_sample_idx = merge_list(s_index[0], m_index[0])
                        total_x_sample = tf.concat((x, x_logit.numpy()[m_index]), axis=0)
                        total_label = tf.concat((y, sample_label[m_index]), axis=0)
                        metrix_list[i]['valid_sample'].append([len(sample_y),
                                                       len(o_sample_y)])
                        metrix_list[i]['total_sample'] = metrix['total_sample'] + list(sample_label)
                        metrix_list[i]['total_valid_sample'] = metrix['total_valid_sample'] + list(sample_y)
                        with tf.GradientTape() as o_tape:
                            _, _, o_loss = compute_loss(model, classifier_list[i], total_x_sample,
                                                        total_label, method=method)
                        o_gradients = o_tape.gradient(o_loss, classifier_list[i].trainable_variables)
                        optimizer.apply_gradients(zip(o_gradients, classifier_list[i].trainable_variables))
                return metrix_list
            else:
                for cls in range(model.num_cls):
                    sample_label = np.array(([cls] * features.shape[0]))
                    z = tf.concat([features, np.expand_dims(sample_label, 1)], axis=1)
                    x_logit = model.sample(z)
                    for i in range(len(classifier_list)):
                        m_index = estimate(classifier, x_logit, model.threshold, sample_label, target)
                        sample_y = sample_label.numpy()[m_index]
                        s_index = estimate(o_classifier, x_logit, model.threshold, sample_label, target)
                        o_sample_y = sample_label.numpy()[s_index]
                        total_sample_idx = merge_list(s_index[0], m_index[0])
                        total_x_sample = x_logit.numpy()[total_sample_idx]
                        total_label = sample_label.numpy()[total_sample_idx]
                        metrix_list[i]['valid_sample'].append([len(sample_y),
                                                               len(o_sample_y)])
                        metrix_list[i]['total_sample'] = metrix['total_sample'] + list(y)
                        metrix_list[i]['total_valid_sample'] = metrix['total_valid_sample'] + list(sample_y)
                        with tf.GradientTape() as o_tape:
                            _, _, o_loss = compute_loss(model, o_classifier, total_x_sample, total_label,
                                                        method=method)
                        o_gradients = o_tape.gradient(o_loss, o_classifier.trainable_variables)
                        optimizer.apply_gradients(zip(o_gradients, o_classifier.trainable_variables))
    e = 0

    for epoch in range(epochs):
        e += 1
        start_time = time.time()
        if (model.data == 'celebA' or "large_celebA"):
            for x, y in tf.data.Dataset.zip((train_set[0], train_set[1])):
                metrix_list = train_step(model, classifier, classifier_list,
                x, y, oversample=True, metrix_list=metrix_list)

        elif (model.data == 'mnist'):
            for x,z,y in tf.data.Dataset.zip((train_set[0], latent, train_set[1])):
                metrix_list = train_step(model, classifier, classifier_list,
                x, y, features=z, oversample=True, metrix_list=metrix_list)


            #generate_and_save_images(model, epochs, r_sample, "rotate_image")
        if (epoch +1)%1 == 0:
            print('*' * 20)
            end_time = time.time()
            print("Epoch: {}, time elapse for current epoch: {}".format(epoch + 1, end_time - start_time))
            ori_loss, h, _ = compute_loss(model, classifier, test_set[0], test_set[1])
            pre_acsa, pre_g_mean, pre_tpr, pre_confMat, pre_acc = indices(h.numpy().argmax(-1), test_set[1])
            for i in range(len(threshold_list)):
                _, o_h, _ = compute_loss(model, classifier_list[i], test_set[0], test_set[1])
                oAsca, oGMean, o_pre_tpr, o_pre_confMat, o_pre_acc = indices(o_h.numpy().argmax(-1), test_set[1])
                elbo = -ori_loss
                pre_train_g_mean_acc = pre_g_mean
                pre_train_acsa_acc = pre_acsa
                o_acsa_acc = oAsca
                o_g_mean_acc = oGMean

                valid_sample = np.array(metrix_list[i]['valid_sample'])
                total_sample = np.array(metrix_list[i]['total_sample'])
                pass_pre_train_classifier = np.sum(valid_sample[:, 0])/len(total_sample.flatten())
                pass_o_classifier = np.sum(valid_sample[:, 1])/len(total_sample.flatten())
                total_valid_sample = np.array(metrix_list[i]['total_valid_sample'])

                ckpt_save_path = checkpoints_list[i].save()
                print('Saving checkpoint at {}'.format(ckpt_save_path))

                result_dir = result_dir_list[i]

                result = {
                    "elbo": elbo,
                    "pre_g_mean": pre_train_g_mean_acc,
                    'pre_acsa': pre_train_acsa_acc,
                    'o_g_mean': o_g_mean_acc,
                    'o_acsa': o_acsa_acc,
                    'pass_pre_train_classifier': pass_pre_train_classifier,
                    'valid_sample': np.sum(valid_sample_num[:, 0]),
                    'pass_o_classifier': pass_o_classifier
                }
                for cls in range(model.num_cls):
                    cls_acc = 'acc_in_cls{}'.format(cls)
                    name = 'valid_ratio_in_cls{}'.format(cls)
                    result[cls_acc] = o_pre_acc[i]
                    valid_sample_num = np.sum(total_valid_sample == cls)
                    total_gen_num = np.sum(total_sample.flatten() == cls)
                    if (valid_sample_num == 0):
                        result[name] = 0
                    else:
                        result[name] = valid_sample_num / total_gen_num
                df = pd.DataFrame(result, index=[e], dtype=np.float32)
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
                if not os.path.isfile(result_dir+'/result.csv'):
                    df.to_csv(result_dir+'/result.csv')
                else:  # else it exists so append without writing the header
                    df.to_csv(result_dir+'/result.csv', mode='a', header=False)


                print('threshld:{} , \n'
                      ' o_g_means:{},  o_acsa:{}, \n'
                      .format(classifier_list[i].threshold,
                              o_g_mean_acc, o_acsa_acc,
                              ))
                print("-" * 20)
            print('*' * 20)
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



