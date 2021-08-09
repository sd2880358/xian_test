from model2 import Classifier
import tensorflow as tf
from dataset import preprocess_images
import numpy as np


optimizer = tf.keras.optimizers.Adam(1e-4)
(train_set, train_labels), (test_set, test_labels) = tf.keras.datasets.mnist.load_data()

date = '8_7'
dims = [9, 9]

classifier = Classifier(shape=[9, 9, 1], num_cls=10)
train_set = preprocess_images(train_set, shape=[28,28,1])
test_set = preprocess_images(test_set, shape=[28,28,1])

train_set = tf.image.resize(train_set, [9,9])
test_set = tf.image.resize(test_set, [9,9])


filePath = "./exhaustion_cls2"
classifier_path = "./checkpoints/" + filePath
cls = tf.train.Checkpoint(classifier=classifier)
cls_manager = tf.train.CheckpointManager(cls, classifier_path, max_to_keep=5)
if cls_manager.latest_checkpoint:
    cls.restore(cls_manager.latest_checkpoint)
    print('classifier checkpoint restored!!')


classifier.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
classifier.fit(train_set, train_labels, epochs=10, verbose=1, shuffle=True,
               batch_size=32, validation_data=(test_set, test_labels))
ckpt_save_path = cls_manager.save()
print('Saving checkpoint for epoch {} at {}'.format(1,
                                                    ckpt_save_path))