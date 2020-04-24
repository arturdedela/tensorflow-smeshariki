from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from manual_dataset_loading import load_data_from_dir

train_images, train_labels = load_data_from_dir('dataset')
test_images, test_labels = load_data_from_dir('dataset-test')

train_images = train_images / 255.0
test_images = test_images / 255.0

print('Total train images: ', len(train_images))

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(100, 100)),
    keras.layers.Dense(80, activation='relu'),
    keras.layers.Dense(5)
])

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()


class TrainCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if not logs:
            return

        if logs.get('accuracy') > 0.90:
            print("\nReached 90% accuracy. Stopping training...")
            self.model.stop_training = True


callback = TrainCallback()

model.fit(train_images, train_labels, epochs=10, callbacks=[callback])

test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Loss: ", test_loss)
print("Accuracy: ", test_acc)

model.save('model')