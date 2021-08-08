import numpy as np
from model2 import Classifier
import tensorflow as tf
from loss import confidence_function
import os
# create a m dimensions list, every s dimension can choose value between l;
def create_list(m, s, l):
    initial_list = np.zeros(m)
    return (flip_number(initial_list, 0, s, l))



# given a m dimension list, current index, and next index, flip value between l,
# return if next index==total length, recursive otherwise;
def flip_number(m, c_i, s, l):
    tmp_list = []
    n_i = c_i + s
    for i in range(len(l)):
        tmp = m.copy()
        tmp[c_i] = l[i]
        tmp_list.append(tmp)

    if (n_i > (len(m) - 1)):
        return tmp_list
    else:
        total_list = []
        for tmp in tmp_list:
            total_list.append(flip_number(tmp, n_i, s, l))
        total_list = np.array(total_list)
        w = total_list.shape[0]
        h = total_list.shape[1]
        return total_list.reshape(w * h, total_list.shape[2])

def initial_dataset(m, s, l, save=False):
    mnist_data = create_list(m, s, l)
    if (save==True):
        np.savez('../dataset/mnist_exhausted_test_initialize',
                mnist_data=mnist_data.reshape([mnist_data.shape[0], 9, 9]))
    return mnist_data.reshape([mnist_data.shape[0], 9, 9, 1])



if __name__ == '__main__':
    os.environ["CUDA_DECICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,4,5,7"
    dataset = initial_dataset(81, 3, [0,1])
    print("dataset has been initial")
    print("the dataset shape is {}".format(dataset.shape))
    classifier = Classifier(shape=[9, 9, 1], num_cls=10)
    checkpoint = tf.train.Checkpoint(classifier=classifier)
    checkpoint.restore("./checkpoints/exhaustion_cls2/ckpt-1")
    threshold = [0.95, 0.95, 0.95, 0.5, 0.8, 0.5, 0.7, 0.95, 0.95, 0.7]
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
