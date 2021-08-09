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
    target = 'margin'
    threshold_list = [0.85, 0.86, 0.87, 0.88, 0.89, 0.9]
    date = '8_8'
    for i in range(1, 11):
        data_name = 'mnist'
        file_path = 'mnist_test{}'.format(i)
        dataset = Dataset(data_name)
        epochs = 30
        method = 'lsq'
        (train_set, train_labels), (test_set, test_labels) = dataset.load_data()
        model = F_VAE(data=data_name, shape=dataset.shape, latent_dim=dataset.latent_dims, model='cnn', num_cls=dataset.num_cls)
        classifier = Classifier(shape=dataset.shape, model='mlp', num_cls=dataset.num_cls)

        checkpoint = tf.train.Checkpoint(sim_clr=model, clssifier=classifier)
        checkpoint.restore("./checkpoints/8_5/pre_train_mnist_lsq/ckpt-40")

        start_train(epochs, target, threshold_list, method, model, classifier, dataset,
                    [train_set, train_labels],
                    [test_set, test_labels], date, file_path)

