import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import scipy.io as sc
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers.experimental import preprocessing
import time
from IPython.display import clear_output
import math
from model import CVAE

def make_classifier():
    model = tf.keras.Sequential()
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(4, activation='relu'))
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(0.01),
                  metrics=['accuracy'])
    return model

cat_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

import math
def find_diff(model, x_1, x_2):
    mean_1, logvar_2 = model.encode(x_1)
    z_1 = model.reparameterize(mean_1, logvar_2).numpy()
    mean_2, logvar_2 = model.encode(x_2)
    z_2 = model.reparameterize(mean_2, logvar_2).numpy()
    diff = np.mean(np.abs(z_1 - z_2), axis=0)
    return diff

def selection(data_labels, labels, label_list, label_name, v_k):
    pre_fixed = labels[labels[label_name] == v_k]
    idx_list = []
    label_list = [i for i in label_list if i != label_name]
    for i in range(len(data_labels)):
        data = data_labels.iloc[i, :]
        sample = pre_fixed.loc[((pre_fixed[label_list[0]] != data[label_list[0]]) &
                        (pre_fixed[label_list[1]] != data[label_list[1]]) &
                        (pre_fixed[label_list[2]] != data[label_list[2]]))
        ].sample(n=1)
        idx_list.append(sample.index)
    return idx_list



def split_label(model, data, labels, split=100):
    label_set = ["scale", "orientation", "x_axis", "y_axis"]
    tmp = []
    features = []
    for i in range(len(label_set)):
        l = len(labels.groupby(label_set[i]).count())
        for j in range(l):
            label = labels[labels[label_set[i]] == j]
            label_idx = label.index
            train_set = data[label_idx]
            subgroups = 1
            for batch in range(subgroups):
                start = batch*split
                end = (batch+1)*split
                x_1 = train_set[start:end]
                x_1_labels = label[start:end]
                x_2_index = selection(x_1_labels, labels, label_set, label_set[i], j)
                x_2 = data[x_2_index].reshape(len(x_1), 64, 64, 1)
                diff = find_diff(model, x_1, x_2)
                features.append(diff)
                tmp.append(i)
    features = np.array(features, dtype="float32")
    tmp = np.array(tmp, dtype='int')
    return features, tmp


if __name__ == '__main__':
    model = CVAE(latent_dim=8, beta=4, shape=[64, 64, 1])
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore("checkpoints/4_25/beta_test/ckpt-31")
    dataset_zip = np.load('../dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')

    imgs = dataset_zip['imgs']
    imgs = np.reshape(imgs, [len(imgs), 64, 64, 1]).astype('float32')
    latents_values = dataset_zip['latents_values']
    latents_classes = dataset_zip['latents_classes']
    latents_classes = pd.DataFrame(latents_classes)
    latents_classes.columns = ["color", "shape", "scale", "orientation", "x_axis", "y_axis"]
    features, labels = split_label(model, imgs, latents_classes)
    train_features = features
    train_labels = labels
    classifer = make_classifier()
    history = classifer.fit(
        train_features, train_labels,
        validation_split=0.2,
        epochs=100, verbose=1)
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    test_results = {}
    hist.to_csv("./score/dis_matrix")