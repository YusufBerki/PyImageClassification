import numpy as np
from PIL import Image
from tensorflow import keras
from skimage import transform

from prediction.src.opt import image_shape


def load_model(path):
    return keras.models.load_model(path)


def load_image(image_path):
    img = Image.open(image_path)
    img = np.array(img).astype('float32') / 255
    img = transform.resize(img, image_shape)
    img = np.expand_dims(img, axis=0)
    return img
