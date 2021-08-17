import tensorflow as tf
from over_sampling import start_train
from dataset import preprocess_images, divide_dataset, imbalance_sample, Dataset
from model import CVAE, Classifier, F_VAE
import os
from celebA import CelebA
from load_data import split
import numpy as np
if __name__ == '__main__':
    os.environ["CUDA_DECICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,4,5,7"

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=8168)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    n = 25
    batch_size = 64
    threshold = [1. , 1., 0.957, 0.973, 0.964, 0.924, 0.927, 0.899, 0.739,0.744]
    #threshold = [0.96, 0.927, 0.899, 0.739, 0.744]
    threshold_list = [threshold]
    date = '8_17'
    epochs = 30
    for i in range(2, 11):
        data_name = 'mnist'
        file_path = 'mnist_super_loss{}'.format(i)
        dataset = Dataset(data_name, batch_size=batch_size)
        method = 'lsq'
        (train_set, train_labels), (test_set, test_labels) = dataset.load_data()
        model = F_VAE(data=data_name, shape=dataset.shape, latent_dim=dataset.latent_dims, model='cnn', num_cls=dataset.num_cls)
        classifier = Classifier(shape=dataset.shape, model='mlp', num_cls=dataset.num_cls)

        checkpoint = tf.train.Checkpoint(sim_clr=model, clssifier=classifier)
        checkpoint.restore("./checkpoints/8_10/pre_train_mnist_super_loss/ckpt-40")

        start_train(epochs, n, threshold_list, method, model, classifier, dataset,
                    [train_set, train_labels],
                    [test_set, test_labels], date, file_path)

