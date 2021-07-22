import os
import numpy as np
# import pandas as pd
import random
# import cv2
# import matplotlib.pyplot as plt

from opt import TRAIN_DATASET_DIR
from data_generator import get_data_generator
from data_loader import get_data_loader
from models.getter import model_getter
from callbacks import get_callbacks

# TODO: Add seed to args
seed = 32
np.random.seed(seed)


def train():
    data_generator = get_data_generator(is_train=True)
    data_generator = get_data_loader(data_generator, TRAIN_DATASET_DIR)
    model = model_getter()

    callbacks = get_callbacks()
    history = model.fit(data_generator, batch_size=16, epochs=10, callbacks=callbacks)


if __name__ == '__main__':
    train()
