from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from opt import CALLBACKS_CONFIG

callbacks = {
    "reduce_lr": ReduceLROnPlateau,
    "model_checkpoint": ModelCheckpoint
}


def get_callbacks():
    return [callbacks[callback](**opt) for callback, opt in CALLBACKS_CONFIG.items()]
