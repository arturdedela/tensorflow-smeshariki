import pathlib

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


data_dir = pathlib.Path('./dataset/')

CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])


def get_datagen():
    # images_paths = list(data_dir.glob('*/*.png'))
    # image_count = len(images_paths)
    # print('Image count: ', image_count)

    image_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    BATCH_SIZE = 32
    IMG_HEIGHT = 150
    IMG_WIDTH = 150
    # STEPS_PER_EPOCH = np.ceil(image_count / BATCH_SIZE)

    train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                         batch_size=BATCH_SIZE,
                                                         shuffle=True,
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         classes=list(CLASS_NAMES),
                                                         color_mode='grayscale')

    return train_data_gen


def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(5, 5, n + 1)
        ax.imshow(image_batch[n])
        plt.title(CLASS_NAMES[label_batch[n] == 1][0].title())
        ax.axis('off')

    plt.show()


# image_batch, label_batch = next(train_data_gen)
# show_batch(image_batch, label_batch)


