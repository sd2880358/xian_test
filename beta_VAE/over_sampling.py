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


def estimate(classifier, x_logit, threshold, label, target):
    conf, l = confidence_function(classifier, x_logit, target=target)
    return np.where((conf>=threshold) & (l==label))

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

def start_train(epochs, target, threshold, model, classifier, o_classifier,
                train_set, test_set, date, filePath):
    sim_optimizer = tf.keras.optimizers.Adam(1e-4)
    cls_optimizer = tf.keras.optimizers.Adam(1e-4)
    o_optimizer = tf.keras.optimizers.Adam(1e-4)
    def train_step(model, classifier, o_classifier, x, y, sim_optimizer,
                   cls_optimizer, oversample=False, threshold=None, metrix=None):
        if (oversample):
            with tf.GradientTape() as o_tape:
                _, _, o_cls_loss = compute_loss(model, o_classifier, x, y)
            o_gradients = o_tape.gradient(o_cls_loss, o_classifier.trainable_variables)
            cls_optimizer.apply_gradients(zip(o_gradients, o_classifier.trainable_variables))
            mean, logvar = model.encode(x)
            features = model.reparameterize(mean, logvar)
            if(model.data=='celebA'):
                for cls in range(model.num_cls):
                    with tf.GradientTape() as o_tape:
                        sample_label = np.array(([cls] * features.shape[0]))
                        z = tf.concat([features, np.expand_dims(sample_label, 1)], axis=1)
                        x_logit = model.sample(z)
                        m_index = estimate(classifier, x_logit, threshold, y, target)
                        sample = x_logit.numpy()[m_index]
                        sample_y = y.numpy()[m_index]
                        ori_loss, h, cls_loss = compute_loss(model, o_classifier, sample, sample_y, gamma=1)
                        s_index = estimate(o_classifier, x_logit, threshold, y, target)
                        o_sample = x_logit.numpy()[s_index]
                        o_sample_y = y.numpy()[s_index]                    
                        _, _, o_loss = compute_loss(model, o_classifier, o_sample, o_sample_y)
                        total_loss = tf.reduce_mean(cls_loss + o_loss)
                        metrix['valid_sample'].append([len(sample_y)/len(sample_label),
                                                       len(o_sample_y)/len(sample_label)])
                        metrix['total_sample'].append([sample_label])
                        metrix['total_valid_sample'].append(merge_list(s_index[0], m_index[0]))
                    o_gradients = o_tape.gradient(total_loss, o_classifier.trainable_variables)
                    o_optimizer.apply_gradients(zip(o_gradients, o_classifier.trainable_variables))
                return metrix
            else:
                r = 5
                n = 10
                triversal_range = np.linspace(-r, r, n)
                for dim in range(features.shape[1]):
                    for replace in triversal_range:
                        with tf.GradientTape() as o_tape:
                            c_features = features.numpy()
                            c_features[:, dim] = replace
                            z = tf.concat([c_features, tf.expand_dims(y, 1)], axis=1)
                            x_logit = model.sample(z)
                            m_index = estimate(classifier, x_logit, threshold, y, target)
                            sample = x_logit.numpy()[m_index]
                            sample_y = y.numpy()[m_index]
                            ori_loss, h, cls_loss = compute_loss(model, o_classifier, sample, sample_y, gamma=1)
                            s_index = estimate(o_classifier, x_logit, threshold, y, target)
                            o_sample = x_logit.numpy()[s_index]
                            o_sample_y = y.numpy()[s_index]
                            _, _, o_loss = compute_loss(model, o_classifier, o_sample, o_sample_y)
                            total_loss = tf.reduce_mean(cls_loss + o_loss)
                            metrix['valid_sample'].append([len(sample_y)/len(y), len(o_sample_y)/len(y)])
                            metrix['total_sample'].append([y])
                            metrix['total_valid_sample'] + list((y[merge_list(s_index[0], m_index[0])]))
                        o_gradients = o_tape.gradient(total_loss, o_classifier.trainable_variables)
                        cls_optimizer.apply_gradients(zip(o_gradients, o_classifier.trainable_variables))
                '''
                sim_gradients = sim_tape.gradient(ori_loss, model.trainable_variables)
                sim_optimizer.apply_gradients(zip(sim_gradients, model.trainable_variables))
                '''
        else:
            with tf.GradientTape() as sim_tape, tf.GradientTape() as cls_tape:
                ori_loss, _, encode_loss = compute_loss(model, classifier, x, y)
            sim_gradients = sim_tape.gradient(ori_loss, model.trainable_variables)
            cls_gradients = cls_tape.gradient(encode_loss, classifier.trainable_variables)
            cls_optimizer.apply_gradients(zip(cls_gradients, classifier.trainable_variables))
            sim_optimizer.apply_gradients(zip(sim_gradients, model.trainable_variables))
    checkpoint_path = "./checkpoints/{}/{}".format(date, filePath)
    ckpt = tf.train.Checkpoint(sim_clr=model,
                               clssifier = classifier,
                               o_classifier=o_classifier,
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
            train_step(model, classifier, o_classifier,
                    x, y, sim_optimizer, cls_optimizer)

        metrix = {}
        metrix['valid_sample'] = []
        metrix['total_sample'] = []
        metrix['total_valid_sample'] = []
        for x, y in tf.data.Dataset.zip((train_set[0], train_set[1])):
            metrix = train_step(model, classifier, o_classifier,
            x, y, sim_optimizer, cls_optimizer, oversample=True, threshold=threshold, metrix=metrix)
        #generate_and_save_images(model, epochs, r_sample, "rotate_image")
        if (epoch +1)%1 == 0:

            valid_sample = np.array(metrix['valid_sample'])
            total_sample = np.array(metrix['total_sample'])
            pass_pre_train_classifier = np.mean(valid_sample[:, 0])
            pass_o_classifier = np.mean(valid_sample[:, 1])
            total_valid_sample = np.array(metrix['total_valid_sample'])


            end_time = time.time()
            elbo_loss = tf.keras.metrics.Mean()
            pre_train_g_mean = tf.keras.metrics.Mean()
            pre_train_acsa = tf.keras.metrics.Mean()
            o_g_mean = tf.keras.metrics.Mean()
            o_acsa = tf.keras.metrics.Mean()
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                        ckpt_save_path))
            for t_x, t_y in tf.data.Dataset.zip((test_set[0], test_set[1])):
                ori_loss, h, _ = compute_loss(model, classifier, t_x, t_y)
                pre_g_mean, pre_acsa = acc_metrix(h.numpy().argmax(-1), t_y.numpy())
                _, o_h, _ = compute_loss(model, o_classifier, t_x, t_y)
                oGMean, oAsca = acc_metrix(o_h.numpy().argmax(-1), t_y.numpy())
                total_loss = ori_loss
                '''
                over_sample_acc(latent_triversal(model, classifier, t_x, t_y, r=3, n=100))
                '''
                pre_train_g_mean(pre_g_mean)
                pre_train_acsa(pre_acsa)
                o_g_mean(oGMean)
                o_acsa(oAsca)
                elbo_loss(total_loss)
            elbo = -elbo_loss.result()
            pre_train_g_mean_acc = pre_train_g_mean.result()
            pre_train_acsa_acc = pre_train_acsa.result()
            o_acsa_acc = o_acsa.result()
            o_g_mean_acc = o_g_mean.result()
            result = {
                "elbo": elbo,
                "pre_g_mean": pre_train_g_mean_acc,
                'pre_acsa': pre_train_acsa_acc,
                'o_g_mean': o_g_mean_acc,
                'o_acsa': o_acsa_acc,
                'pass_pre_train_classifier': pass_pre_train_classifier,
                'pass_o_classifier': pass_o_classifier
            }

            for i in range(model.num_cls):
                name = 'cls{}'.format(i)
                valid_sample_num = np.sum(total_valid_sample == i)
                total_gen_num = np.sum(total_sample == i)
                result[name] = valid_sample_num / total_gen_num
            df = pd.DataFrame(result, index=[e], dtype=np.float32)
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            if not os.path.isfile(result_dir+'/result.csv'):
                df.to_csv(result_dir+'/result.csv')
            else:  # else it exists so append without writing the header
                df.to_csv(result_dir+'/result.csv', mode='a', header=False)
            print('*' * 20)
            print('Epoch: {}, elbo: {}, \n'
                  ' pre_g_means: {}, pre_acsa: {}, \n, o_g_means:{},  o_acsa:{}, \n' 
                  'time elapse for current epoch: {}'
                  .format(epoch+1, elbo,pre_train_g_mean_acc,
                          pre_train_acsa_acc, o_g_mean_acc, o_acsa_acc,
                          end_time - start_time))
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



