from options.train_options import TrainOptions

from data.data_generator import get_data_generator
from data.data_loader import get_data_loader

from models.getter import model_getter

from utils.callbacks import get_callbacks

import warnings

warnings.filterwarnings("ignore")


def train(opt):
    data_generator = get_data_generator(opt)
    data_generator = get_data_loader(opt, data_generator, opt.train_dataset_dir)
    model = model_getter(opt)

    callbacks = get_callbacks(opt)

    history = model.fit(data_generator, batch_size=16, epochs=10, callbacks=callbacks)


if __name__ == '__main__':
    opt = TrainOptions().parse()

    train(opt)
