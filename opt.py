import os
from datetime import datetime

now = datetime.now().strftime('%Y_%m_%d_%H_%M')


def makedirs(path):
    dir_name, _ = os.path.split(path)
    os.makedirs(dir_name, exist_ok=True)
    return path


BASE_DIR = os.getcwd()

# Dataset
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
TRAIN_DATASET_DIR = os.path.join(DATASET_DIR, 'train')
TEST_DATASET_DIR = os.path.join(DATASET_DIR, 'test')

# Checkpoints Dir
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')

# Data Generator Config
DATA_GENERATOR_CONFIG = {
    "horizontal_flip": True,
    "vertical_flip": True,
    "zoom_range": 0.4,
}

# Data Loader Config
DATA_LOADER_CONFIG = {
    "target_size": (150, 150),
    "batch_size": 16,
    "shuffle": True
}

# Model Parameters
MODEL_PARAMS = {
    "model": "efficient_net_b5",
    "params": {
        "weights": "imagenet",
        "include_top": False,
        "input_shape": (160, 160, 3)
    },
    "number_of_classes": len(os.listdir(TRAIN_DATASET_DIR)),
    "activation": 'softmax',
    "loss": "categorical_crossentropy",
    "optimizer": "adam",
    "metrics": ["accuracy"]
}

# Callbacks Config
CALLBACKS_CONFIG = {
    "reduce_lr": {
        "monitor": "loss",
        "factor": 0.3,
        "patience": 5,
        "verbose": 2,
        "mode": "max"
    },
    "model_checkpoint": {
        "filepath": makedirs(os.path.join(CHECKPOINT_DIR, f"{MODEL_PARAMS['model']}_{now}", "epoch_{epoch:02d}.h5")),
        "save_weights_only": True
    },
}
