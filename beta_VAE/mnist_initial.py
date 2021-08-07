import numpy as np
from model2 import Classifier
import tensorflow as tf
from loss import confidence_function

# create a m dimensions list, every s dimension can choose value between l;
def create_list(m, s, l):
    d = {}
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

def initial_dataset():
    l = [0, 1]
    mnist_data = create_list(81, 3, l)
    np.savez('../dataset/mnist_exhausted_test_initialize',
             mnist_data=mnist_data.reshape([mnist_data[0], 9, 9]))

if __name__ == '__main__':
    initial_dataset()
    print("dataset has been initial")
    file = np.load("../dataset/mnist_exhausted_test_initialize.npz")
    dataset = file['mnist_dataset']
    classifier = Classifier(shape=[9, 9, 1], num_cls=10)
    checkpoint = tf.train.Checkpoint(classifier=classifier)
    checkpoint.restore("./checkpoints/exhausion_cls2/ckpt-1")
    conf, l = confidence_function(classifier, dataset, target='margin')
    threshold = 0.95
    tmp_data_list = []
    tmp_label_list = []
    num = 0
    for i in range(10):
        tmp_data = dataset(np.where((conf.numpy()>=threshold) & (l==i)))
        tmp_label = np.array([i]*len(tmp_data))
        tmp_data_list.append(tmp_data)
        tmp_label_list.append(tmp_label)
        num += len(tmp_data)

    valid_data = np.zeros([num, 9, 9 ,1])
    valid_label = np.zeros([num,])
    idx = 0
    for i in range(10):
        l = len(tmp_data_list[i])
        tmp_data[idx : idx+l, :, :, :] = tmp_data_list[i]
        tmp_label[idx : idx + l] = tmp_label_list[i]
        idx += l
    np.savez("./mnist_exhauster_test_data.npz", mnist_data=valid_data, mnist_labels=valid_label)
