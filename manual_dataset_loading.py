import os
import pathlib

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def load_data_from_dir(dir_path):
    data_dir = pathlib.Path(dir_path)
    images_paths = list(data_dir.glob('*/*.png'))

    CLASS_NAMES = list([item.name for item in data_dir.glob('*')])

    train_images = list()
    train_labels = list()

    for path in images_paths:
        img: Image.Image = Image.open(path)
        img = img.resize((200, 200))
        img = np.asarray(img)

        parts = str(path).split(os.path.sep)
        label = CLASS_NAMES.index(parts[-2])

        train_images.append(img)
        train_labels.append(label)

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    return train_images, train_labels
