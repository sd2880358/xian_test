import os
import numpy as np
from beta_VAE.celebA import CelebA
import pandas as pd


def load_celeba(path):
    data = np.load(os.path.join(path, "data.npy"))
    data = data.astype(float)
    data = data / 255.0 #0~1
    return data

def split(path):

    celebA = CelebA(
        main_folder = path,
        drop_features=[
        'Attractive',
        'Pale_Skin',
        'Blurry',
    ])

    dataset = load_celeba(path)
    features = pd.read_csv(os.path.join(path, "label_set.csv"), index_col='index')

    train_set = celebA.split()
    train_set = train_set[train_set.index.isin(features.index)]

    test_set = celebA.split('test')
    test_set = test_set[test_set.index.isin(features.index)]
    labels = features['hair_style']

    train_labels = labels[train_set.index]
    test_labels = labels[test_set.index]
    return (dataset[train_set.index], train_labels.to_numpy()), (dataset[test_set.index], test_labels.to_numpy())



if __name__ == "__main__":
    test = load_celeba("../CelebA/")
    print(len(test))