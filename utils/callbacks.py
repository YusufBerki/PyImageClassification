from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from utils.utils import get_suffix

callbacks = {
    "reduce_lr": ReduceLROnPlateau,
    "model_checkpoint": ModelCheckpoint
}


def get_callbacks(opt):
    opt = vars(opt)
    return [func(**get_suffix(opt, f'{callback}_')) for callback, func in callbacks.items() if callback in opt.keys() and opt[callback]]
