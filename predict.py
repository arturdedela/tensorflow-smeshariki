import pathlib

from tensorflow import keras
from PIL import Image
import numpy as np

data_dir = pathlib.Path('dataset')
CLASS_NAMES = list([item.name for item in data_dir.glob('*')])

model: keras.models.Sequential = keras.models.load_model('model')

img: Image.Image = Image.open('./predict-examples/krosh.png')
img = img.resize((200, 200))
img = np.asarray(img) / 255


prediction = model.predict(img)

print(prediction)
