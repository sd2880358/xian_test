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
    threshold_list = [0.95, 0.96, 0.97, 0.98, 0.99, 1]
    date = '8_7'
    data_name = 'large_celebA'
    file_path = 'large_celebA_test'
    dataset = Dataset(data_name)
    epochs = 30
    method = 'lsq'
    (train_set, train_labels), (test_set, test_labels) = dataset.load_data()
    model = F_VAE(data=data_name, shape=dataset.shape, latent_dim=dataset.latent_dims, model='cnn', num_cls=dataset.num_cls)
    classifier = Classifier(shape=dataset.shape, model='mlp', num_cls=dataset.num_cls)

    checkpoint = tf.train.Checkpoint(sim_clr=model, clssifier=classifier)
    checkpoint.restore("./checkpoints/8_5/pre_train_large_celebA_lsq/ckpt-20")

    start_train(epochs, target, threshold_list, method, model, classifier, dataset,
                [train_set, train_labels],
                [test_set, test_labels], date, file_path)

