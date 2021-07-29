from utils.utils import get_suffix
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)

callbacks = {
    "reduce_lr": ReduceLROnPlateau,
    "model_checkpoint": ModelCheckpoint,
    "early_stopping": EarlyStopping,
    "tensorboard":TensorBoard
}


def get_callbacks(opt):
    opt = vars(opt)
    return [func(**get_suffix(opt, f'{callback}_')) for callback, func in callbacks.items() if callback in opt.keys() and opt[callback]]
