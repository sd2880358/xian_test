import numpy as np
from model2 import Classifier
from model import CVAE, F_VAE
import tensorflow as tf
from loss import confidence_function, compute_loss, indices
from dataset import preprocess_images
import os
import time
import pandas as pd
# create a m dimensions list, every s dimension can choose value between l;

def start_train(epochs, c_epochs, model, classifier, method,
                train_set, test_set, date, filePath):
    sim_optimizer = tf.keras.optimizers.Adam(1e-4)
    cls_optimizer = tf.keras.optimizers.Adam(1e-4)
    def train_step(model, classifier, x, y, epoch, sim_optimizer,
                   cls_optimizer):
            with tf.GradientTape() as sim_tape, tf.GradientTape() as cls_tape:
                ori_loss, _, encode_loss = compute_loss(model, classifier, x, y, method=method)
            sim_gradients = sim_tape.gradient(ori_loss, model.trainable_variables)
            sim_optimizer.apply_gradients(zip(sim_gradients, model.trainable_variables))
            if (epoch < c_epochs):
                cls_gradients = cls_tape.gradient(encode_loss, classifier.trainable_variables)
                cls_optimizer.apply_gradients(zip(cls_gradients, classifier.trainable_variables))
    checkpoint_path = "./checkpoints/{}/{}".format(date, filePath)
    ckpt = tf.train.Checkpoint(sim_clr=model,
                               clssifier=classifier,
                               optimizer=sim_optimizer,
                               cls_optimizer=cls_optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    result_dir = "./score/{}/{}".format(date, filePath)
    if os.path.isfile(result_dir+'/result.csv'):
        e = pd.read_csv(result_dir+'/result.csv').index[-1]
    else:
        e = 0
    for epoch in range(epochs):
        e += 1
        start_time = time.time()
        for x, y in tf.data.Dataset.zip((train_set[0], train_set[1])):
            train_step(model, classifier, x, y, epoch, sim_optimizer, cls_optimizer)

        #generate_and_save_images(model, epochs, r_sample, "rotate_image")
        if (epoch +1)%5 == 0:

            end_time = time.time()
            elbo_loss = tf.keras.metrics.Mean()
            pre_train_g_mean = tf.keras.metrics.Mean()
            pre_train_acsa = tf.keras.metrics.Mean()
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                        ckpt_save_path))
            ori_loss, h, _ = compute_loss(model, classifier, test_set[0], test_set[1])
            pre_acsa, pre_g_mean, pre_tpr, pre_confMat, pre_acc = indices(h.numpy().argmax(-1), test_set[1])
            total_loss = ori_loss

            pre_train_g_mean(pre_g_mean)
            pre_train_acsa(pre_acsa)
            elbo_loss(total_loss)
            elbo = -elbo_loss.result()
            pre_train_g_mean_acc = pre_train_g_mean.result()
            pre_train_acsa_acc = pre_train_acsa.result()


            result = {
                "elbo": elbo,
                "pre_g_mean": pre_train_g_mean_acc,
                'pre_acsa': pre_train_acsa_acc
            }
            df = pd.DataFrame(result, index=[e], dtype=np.float32)
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            if not os.path.isfile(result_dir+'/result.csv'):
                df.to_csv(result_dir+'/result.csv')
            else:  # else it exists so append without writing the header
                df.to_csv(result_dir+'/result.csv', mode='a', header=False)

            print('*' * 20)
            print('Epoch: {}, elbo: {}, \n'
                  ' pre_g_means: {}, pre_acsa: {}, \n,'
                  'time elapse for current epoch: {}'
                  .format(epoch+1, elbo,pre_train_g_mean_acc,
                          pre_train_acsa_acc,
                          end_time - start_time))
            print('*' * 20)
    #compute_and_save_inception_score(model, file_path)




# given a m dimension list, current index, and next index, flip value between l,
# return if next index==total length, recursive otherwise;
def create_list(m, idx, l):
    initial_list = np.zeros(m)
    return (flip_number(initial_list, idx, l))

def flip_number(m, idx, l):
    tmp_list = []
    c_i = idx[0]
    for i in range(len(l)):
        tmp = m.copy()
        tmp[c_i] = l[i]
        tmp_list.append(tmp)
    if (len(idx[1:]) < 1):
        return tmp_list
    else:
        total_list = []
        for tmp in tmp_list:
            total_list.append(flip_number(tmp, idx[1:], l))
        total_list = np.array(total_list)
        w = total_list.shape[0]
        h = total_list.shape[1]
        return total_list.reshape(w * h, total_list.shape[2])

def initial_dataset(m, idx, l, save=False):
    mnist_data = create_list(m, idx, l)
    if (save==True):
        np.savez('../dataset/mnist_exhausted_test_initialize',
                mnist_data=mnist_data.reshape([mnist_data.shape[0], 9, 9]))
    return mnist_data.reshape([mnist_data.shape[0], 9, 9, 1])

def exhaustion_initialized():
    os.environ["CUDA_DECICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,4,5,7"
    idx = np.random.randint(low=0, high=81, size=27)
    dataset = initial_dataset(81, idx, [0,1])
    print("dataset has been initial")
    print("the dataset shape is {}".format(dataset.shape))
    classifier = Classifier(shape=[9, 9, 1], num_cls=10)
    checkpoint = tf.train.Checkpoint(classifier=classifier)
    checkpoint.restore("./checkpoints/exhaustion_cls2/ckpt-1")
    threshold = [0.95, 0.99, 0.99, 0.99, 0.97, 0.99, 0.98, 0.98, 0.96, 0.95]
    tmp_data_list = []
    tmp_label_list = []
    num = 0
    total_length = dataset.shape[0]
    split_size = 100000
    split = int(np.ceil(total_length / split_size))
    s_idx = 0
    for _ in range(split):
        e_idx = s_idx + split_size
        test_data = dataset[s_idx : e_idx]
        conf, l = confidence_function(classifier, test_data)
        s_idx += split_size
        for i in range(len(threshold)):
            tmp_data = test_data[np.where((conf.numpy()>=threshold[i]) & (l==i))]
            tmp_label = np.array([i]*len(tmp_data))
            tmp_data_list.append(tmp_data)
            tmp_label_list.append(tmp_label)
            num += len(tmp_data)
    print(num)
    print('data has been classified!')
    valid_data = np.zeros([num, 9, 9 ,1])
    valid_label = np.zeros([num,])
    idx = 0
    for i in range(len(tmp_data_list)):
        l = tmp_data_list[i].shape[0]
        valid_data[idx : idx + l, :, :, :] = tmp_data_list[i]
        valid_label[idx : idx + l] = tmp_label_list[i]
        idx += l
    np.savez("../dataset/mnist_exhaustion_test_data.npz", mnist_data=valid_data.astype('float32'),
             mnist_labels=valid_label.astype('int32'))

def model_initialize():
    (train_set, train_labels), (test_set, test_labels) = tf.keras.datasets.mnist.load_data()
    classifier = Classifier(shape=[9, 9, 1], num_cls=10)
    train_set = preprocess_images(train_set, shape=[28, 28, 1])
    test_set = preprocess_images(test_set, shape=[28, 28, 1])
    epochs = 200
    c_epochs = 30
    method = 'lsq'
    file_path = 'beta_VAE_exhaustion_test'
    train_set = tf.image.resize(train_set, [9, 9])
    test_set = tf.image.resize(test_set, [9, 9])
    sim_clr = F_VAE(data='mnist', shape=[9,9,1], latent_dim=4, model='mlp', num_cls=10)
    date = '8_7'
    start_train(epochs, c_epochs, sim_clr, classifier, method,
                [train_set, train_labels],
                [test_set, test_labels], date, file_path)



if __name__ == '__main__':

    exhaustion_initialized()
