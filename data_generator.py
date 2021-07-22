from keras.preprocessing.image import ImageDataGenerator
from opt import DATA_GENERATOR_CONFIG


def get_data_generator(is_train):
    if is_train:
        return ImageDataGenerator(**DATA_GENERATOR_CONFIG)
    return ImageDataGenerator()
