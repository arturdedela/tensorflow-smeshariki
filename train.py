from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from manual_dataset_loading import load_data_from_dir

train_images, train_labels = load_data_from_dir('dataset')
test_images, test_labels = load_data_from_dir('dataset-test')

print('Total train images: ', len(train_images))

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(200, 200)),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(4)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()


class TrainCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if not logs:
            return

        if logs.get('accuracy') > 0.90:
            print("\nReached 98% accuracy. Stopping training...")
            self.model.stop_training = True


callback = TrainCallback()

model.fit(train_images, train_labels, epochs=10, callbacks=[callback])

# test_loss, test_acc = model.evaluate(test_images, test_labels)
#
# print("Loss: ", test_loss)
# print("Accuracy: ", test_acc)

model.save('model')

# model.fit(labeled_ds, epochs=3)
# data_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
#
# BATCH_SIZE = 32
# IMG_HEIGHT = 100
# IMG_WIDTH = 100
# STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)
#
# train_data_get = data_generator.flow_from_directory(directory=str(data_dir),
#                                                     batch_size=BATCH_SIZE,
#                                                     shuffle=True,
#                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
#                                                     classes=list(CLASS_NAMES))
#
#
# def show_batch(image_batch, label_batch):
#     pyplot.figure(figsize=(10, 10))
#     for n in range(25):
#         ax = pyplot.subplot(5, 5, n + 1)
#         ax.imshow(image_batch[n])
#         pyplot.title(CLASS_NAMES[label_batch[n] == 1][0].title())
#         ax.axis('off')
#     pyplot.show()
#
#
# image_batch, label_batch = next(train_data_get)
# print('Image batch: ', image_batch)
# print('Label batch: ', label_batch)
# show_batch(image_batch, label_batch)


# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(100, 100, 3)),
#     keras.layers.Dense(256, activation=nn.relu),
#     keras.layers.Dense(2, activation=nn.softmax)
# ])
#
# model.compile(optimizer=keras.optimizers.Adam(),
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

#
# class TrainCallback(keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         if not logs:
#             return
#
#         if logs.get('accuracy') > 0.9:
#             print("\nReached 90% accuracy. Stopping training...")
#             self.model.stop_training = True
#
#
# callback = TrainCallback()
#
# model.summary()
#
# model.fit(train_data_get, epochs=1)
#
# test_loss, test_acc = model.evaluate(test_images, test_labels)
#
# print("Loss: ", test_loss)
# print("Accuracy: ", test_acc)
