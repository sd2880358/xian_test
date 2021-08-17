from model import Classifier
from dataset import preprocess_images, rotate_dataset, imbalance_sample, Dataset
from loss import indices
import numpy as np
import os
import tensorflow as tf

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

    data_name = 'mnist'
    batch_size = 32
    dataset = Dataset(data_name, batch_size=batch_size)
    method = 'lsq'
    date = '8_17'
    (train_set, train_labels), (test_set, test_labels) = dataset.load_data(normalize=False)
    classifier = Classifier(shape=dataset.shape, model='mlp', num_cls=dataset.num_cls)
    for i in range(10):
        filePath = "./mnist_base_line{}/{}/".format(date, i)
        classifier_path = "./checkpoints/" + filePath
        cls = tf.train.Checkpoint(classifier=classifier)
        cls_manager = tf.train.CheckpointManager(cls, classifier_path, max_to_keep=5)
        if cls_manager.latest_checkpoint:
            cls.restore(cls_manager.latest_checkpoint)
            print('classifier checkpoint restored!!')

        classifier.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        classifier.fit(train_set, train_labels, epochs=150, verbose=1, shuffle=True,
                       batch_size=32,
                       validation_data=(test_set, test_labels))
        prediction = classifier.call(test_set)
        Asca, GMean, tpr, confMat, acc = indices(prediction.numpy().argmax(-1), test_labels)
        print('\nTest acsa:{}, Test GMeans:{}'.format(Asca, GMean))
        ckpt_save_path = cls_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(1,
                                                            ckpt_save_path))