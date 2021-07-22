import os
import json
import codecs
import numpy as np
from tensorflow import keras
from datetime import datetime

# Some models (eg EfficientNet) need to be imported before they can be installed.
from models.getter import *


def get_now():
    return datetime.now().strftime('%Y_%m_%d_%H_%M')


def makedirs(path):
    dir_name, _ = os.path.split(path)
    os.makedirs(dir_name, exist_ok=True)
    return path


def makedir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def fix_opt_paths(opt):
    opt.train_dataset_dir = os.path.join(opt.dataset_path, opt.train_dataset_dir)
    opt.test_dataset_dir = os.path.join(opt.dataset_path, opt.test_dataset_dir)
    return opt


def get_suffix(opt, prefix):
    if not isinstance(opt, dict):
        opt = vars(opt)

    return {k.replace(prefix, ''): v for k, v in opt.items() if k.startswith(prefix) and k != prefix}


def save_history(path, history):
    makedirs(path)
    new_hist = {}
    for key in list(history.history.keys()):
        new_hist[key] = history.history[key]
        if type(history.history[key]) == np.ndarray:
            new_hist[key] = history.history[key].tolist()
        elif type(history.history[key]) == list:
            if type(history.history[key][0]) == np.float64:
                new_hist[key] = list(map(float, history.history[key]))

    with codecs.open(path, 'w', encoding='utf-8') as file:
        json.dump(new_hist, file, separators=(',', ':'), sort_keys=True, indent=4)


def load_history(path):
    with codecs.open(path, 'r', encoding='utf-8') as file:
        n = json.loads(file.read())
    return n


def save_model(path, model):
    makedirs(path)
    model.save(path, save_format='h5')


def load_model(path):
    return keras.models.load_model(path)
