import numpy as np
from tensorflow_addons.image import rotate
import pandas as pd
from tensorflow.keras import datasets
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

def load_data(name):
  switcher = {
    'mnist': datasets.mnist.load_data(),
    'fashion_mnist': datasets.fashion_mnist.load_data(),
    'cifar_10': datasets.svhn_cropped.load_data(),
  }