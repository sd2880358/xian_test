from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Dense, BatchNormalization
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from load_data import load_celeba
from celebA import CelebA
import numpy as np
import tensorflow as tf
from math import ceil, floor
from scipy.linalg import sqrtm

def build_model(num_features):
    base = MobileNetV2(input_shape=(32, 32, 3),
                       weights=None,
                       include_top=False,
                       pooling='avg')  # GlobalAveragePooling 2D

    x = base.output
    x = Dense(1536, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    top = Dense(num_features, activation='sigmoid')(x)
    return Model(inputs=base.input, outputs=top)

def inception_score(p_yx, eps):
    p_y = np.expand_dims(p_yx.mean(axis=0), 0)
    kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
    sum_kl_d = kl_d.sum(axis=1)
    avg_kl_d = np.mean(sum_kl_d)
    is_score = np.exp(avg_kl_d)
    return ceil(is_score)



def calculate_fid(real, fake):
    mu1, sigma1 = real.mean(axis=0), np.cov(real, rowvar=False)
    mu2, sigma2 = fake.mean(axis=0), np.cov(fake, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def compute_score(X, Y, n_split=10, eps=1E-16):
    model = build_model(37)
    checkpoint_path = "./checkpoints/celebA"
    cls = tf.train.Checkpoint(model=model)
    cls_manager = tf.train.CheckpointManager(cls, checkpoint_path, max_to_keep=5)
    if cls_manager.latest_checkpoint:
        cls.restore(cls_manager.latest_checkpoint)
    prediction = model.predict(X)
    actual = model.predict(Y)
    fid = calculate_fid(prediction, actual)
    score_list = []
    n_part = floor(prediction.shape[0] / n_split)
    for i in range(n_split):
        ix_start, ix_end = i * n_part, (i + 1) * n_part
        subset_X = prediction[ix_start:ix_end, :]
        score_list.append(inception_score(subset_X, eps))
    is_avg, is_std = np.mean(score_list), np.std(score_list)
    return fid

if __name__ == '__main__':
    dataset = load_celeba("../CelebA/")
    batch_size = 60
    epochs = 20
    celeba = CelebA(drop_features=[
    'Attractive',
    'Pale_Skin',
    'Blurry',
    ])
    train_datagen = ImageDataGenerator(rotation_range=100,
                                       rescale=1./255,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest')
    valid_datagen = ImageDataGenerator(rescale=1./255)
    train_split = celeba.split('training'  , drop_zero=False)
    valid_split = celeba.split('validation', drop_zero=False)
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_split,
        directory=celeba.images_folder,
        x_col='image_id',
        y_col=celeba.features_name,
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode='other'
    )
    valid_generator = valid_datagen.flow_from_dataframe(
        dataframe=valid_split,
        directory=celeba.images_folder,
        x_col='image_id',
        y_col=celeba.features_name,
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode='other'
    )
    model = build_model(num_features=celeba.num_features)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics='binary_accuracy')
    filePath = "./celebA"
    checkpoint_path = "./checkpoints/" + filePath
    ckpt = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('classifier checkpoint restored!!')
    print(len(train_generator))
    history = model.fit_generator(
        train_generator,
        epochs=epochs,
        steps_per_epoch=len(train_generator),
        validation_data=valid_generator,
        validation_steps=len(valid_generator),
        max_queue_size=1,
        shuffle=True,
        verbose=1)
    ckpt_save_path = ckpt_manager.save()
    print('Saving checkpoint for epoch {} at {}'.format(1,
                                                        ckpt_save_path))