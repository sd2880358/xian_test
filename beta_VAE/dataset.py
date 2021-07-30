import numpy as np
from tensorflow_addons.image import rotate
import pandas as pd
from tensorflow.keras import datasets
import tensorflow as tf
from load_data import split
def preprocess_images(images, shape):
  images = images.reshape((images.shape[0], shape[0], shape[1], shape[2])) / 255.
  return np.where(images > .5, 1.0, 0.0).astype('float32')

def divide_dataset(train_data, train_labels, sample_size):
  labels = pd.DataFrame({'labels': train_labels})
  dataset = []
  for i in range(0, 10):
    idx = labels[labels.labels == i].iloc[:sample_size].index
    train_images = train_data[idx]
    dataset.append(train_images)
  return np.array(dataset).reshape(10 * sample_size, 28, 28, 1)

def rotate_dataset(image, label, rotate_set):
  s = rotate_set[0]
  e = rotate_set[1]
  dataset = image
  labelset = label
  for degree in range(s+10, e+10, 10):
    d = np.radians(degree)
    r_image = rotate(image, d)
    dataset = np.concatenate([r_image, dataset])
    labelset = np.concatenate([labelset, label])
  return dataset, labelset


def imbalance_sample(data, labels, irs):
  dataset = np.zeros([sum(irs), data.shape[1], data.shape[2], data.shape[3]]).astype('float32')
  label_set = np.zeros([sum(irs)], dtype=int)
  s = 0
  for i in range(len(irs)):
    tmp_data = data[np.where(labels == i)][:irs[i], :, :]
    dataset[s:s + irs[i], :, :, :] = tmp_data
    label_set[s:s + irs[i]] = [i] * irs[i]
    s += irs[i]
  return dataset, label_set




class Dataset():

  def __init__(self, dataset, batch_size=32):
    self.batch_size = batch_size
    self.dataset = dataset
    self.switcher = {
      'mnist': datasets.mnist.load_data()
      #'celebA': split('../CelebA')
    }

    if (dataset == 'mnist'):
      self.shape = (28, 28, 1)
      self.num_cls = 10
      self.latent_dims = 8
      self.irs = [4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40]

    elif (dataset == 'celebA'):
      self.shape = (64, 64, 3)
      self.num_cls = 5
      self.latent_dims = 64
      self.irs = [15000, 1500, 750, 300, 150]

  def load_data(self):
    (train_set, train_labels), (test_set, test_labels) = self.switcher[self.dataset]

    train_set = preprocess_images(train_set, shape=self.shape)
    test_set = preprocess_images(test_set, shape=self.shape)



    train_images, train_labels = imbalance_sample(train_set, train_labels, self.irs)
    train_images = (tf.data.Dataset.from_tensor_slices(train_images)
                    .shuffle(len(train_images), seed=1).batch(self.batch_size))

    train_labels = (tf.data.Dataset.from_tensor_slices(train_labels)
                    .shuffle(len(train_labels), seed=1).batch(self.batch_size))

    test_images = (tf.data.Dataset.from_tensor_slices(test_set)
                   .shuffle(len(test_set), seed=1).batch(self.batch_size))

    test_labels = (tf.data.Dataset.from_tensor_slices(test_labels)
                      .shuffle(len(test_labels), seed=1).batch(self.batch_size))
    return (train_images, train_labels), (test_images, test_labels)