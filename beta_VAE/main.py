import tensorflow as tf
from over_sampling import start_train
from dataset import preprocess_images, divide_dataset, imbalance_sample, Dataset
from model import CVAE, Classifier, F_VAE
from celebA import CelebA
from load_data import split
import numpy as np
if __name__ == '__main__':
    target = 'margin'
    threshold = 0.995
    date = '8_3'
    data_name = 'celebA'
    file_path = 'celebA_test4'
    dataset = Dataset(data_name)
    epochs = 100
    (train_set, train_labels), (test_set, test_labels) = dataset.load_data()
    model = F_VAE(data=data_name, shape=dataset.shape, latent_dim=dataset.latent_dims, model='cnn', num_cls=dataset.num_cls)
    classifier = Classifier(shape=dataset.shape, model='mlp', num_cls=dataset.num_cls)
    o_classifier = Classifier(shape=dataset.shape, model='mlp', num_cls=dataset.num_cls)

    checkpoint = tf.train.Checkpoint(sim_clr=model, clssifier=classifier)
    checkpoint.restore("./checkpoints/8_3/pre_train_celebA/ckpt-42")

    start_train(epochs, target, threshold, model, classifier, o_classifier,
                [train_set, train_labels],
                [test_set, test_labels], date, file_path)