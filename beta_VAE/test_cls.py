from model import Div_Com
import tensorflow as tf
from tensorflow_addons.image import rotate
import time
import matplotlib.pyplot as plt
import numpy as np
import os
from IPython import display
import math
import pandas as pd


optimizer = tf.keras.optimizers.Adam(1e-4)
label_index = ['device_type', 'application', 'mobility']
num_class_index = [3, 4, 3]
date = '7_12'
dims = [120, 168]

classifier = Div_Com(shape=[dims[1], 6, 1], num_cls=num_class_index[0], model='cnn')

file = np.load('../communication_data/dataset{}.npz'.format(1))
dataset = file['dataset']
labelset = file['labelset'][:, 0]

filePath = "./test_classification1"
classifier_path = "./checkpoints/" + filePath
cls = tf.train.Checkpoint(classifier=classifier)
cls_manager = tf.train.CheckpointManager(cls, classifier_path, max_to_keep=5)
if cls_manager.latest_checkpoint:
    cls.restore(cls_manager.latest_checkpoint)
    print('classifier checkpoint restored!!')


classifier.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
classifier.fit(dataset, labelset, epochs=100, verbose=1, shuffle=True,
               batch_size=32, validation_split=0.1)
ckpt_save_path = cls_manager.save()
print('Saving checkpoint for epoch {} at {}'.format(1,
                                                    ckpt_save_path))