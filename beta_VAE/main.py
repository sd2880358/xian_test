import tensorflow as tf
from over_sampling import start_train
from dataset import preprocess_images, divide_dataset, imbalance_sample, Dataset
from model import CVAE, Classifier, F_VAE
from celebA import CelebA
from load_data import split
import numpy as np
if __name__ == '__main__':
    target = 'margin'
    threshold = 0.85
    date = '7_29'
    data_name = 'celebA'
    file_path = data_name + "1"
    dataset = Dataset(data_name)
    epochs = 100
    (train_set, train_labels), (test_set, test_labels) = dataset.load_data()
    sim_clr = F_VAE(data=data_name, shape=dataset.shape, latent_dim=dataset.latent_dims, model='cnn', num_cls=dataset.num_cls)
    classifier = Classifier(shape=dataset.shape, model='cnn', num_cls=dataset.num_cls)
    o_classifier = Classifier(shape=dataset.shape, model='cnn', num_cls=dataset.num_cls)

    '''
    epochs = 200
    batch_size = 32
    shape = [32, 32, 3]
    latent_dim = 64
    num_cls = 4
    (train_set, train_labels), (test_dataset, test_labels) = split(path='../CelebA/')
    #irs = [4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40]
    irs = [15000, 1500, 750, 300, 150]
    train_labels, test_labels = train_labels.flatten(), test_labels.flatten()
    majority_images = train_set[np.where(train_labels == 0)][:irs[0]]

    majority_labels = [0] * irs[0]
    train_set, train_labels = imbalance_sample(train_set, train_labels, irs)
    

    train_set = (tf.data.Dataset.from_tensor_slices(train_set)
                    .shuffle(len(train_set), seed=1).batch(batch_size))

    train_labels = (tf.data.Dataset.from_tensor_slices(train_labels)
                    .shuffle(len(train_labels), seed=1).batch(batch_size))

    majority_images = (tf.data.Dataset.from_tensor_slices(majority_images)
                       .shuffle(len(majority_images), seed=2).batch(batch_size))

    majority_labels = (tf.data.Dataset.from_tensor_slices(majority_labels)
                       .shuffle(len(majority_labels), seed=2).batch(batch_size))

    test_dataset = (tf.data.Dataset.from_tensor_slices(test_dataset)
                   .shuffle(len(test_dataset), seed=1).batch(batch_size))

    test_labels = (tf.data.Dataset.from_tensor_slices(test_labels)
                      .shuffle(len(test_labels), seed=1).batch(batch_size))
    '''
    start_train(epochs, target, threshold, sim_clr, classifier, o_classifier,
                [train_set, train_labels],
                [test_set, test_labels], date, file_path)