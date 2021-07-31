import os
import warnings

from models.getter import model_getter
from options.train_options import TrainOptions

from data.data_loader import get_data_loader
from data.data_generator import get_data_generator

from utils.utils import save_history, save_model
from utils.callbacks import get_callbacks

warnings.filterwarnings("ignore")


def train(opt):
    # Get data generator
    data_generator = get_data_generator(opt)

    # Load data
    data_generator = get_data_loader(opt, data_generator, opt.train_dataset_dir)

    # Get model
    model = model_getter(opt)

    # Get callbacks
    callbacks = get_callbacks(opt)

    # Fit model
    history = model.fit(data_generator, batch_size=16, epochs=10, callbacks=callbacks)

    # Save history
    history_path = os.path.join(opt.results_dir, 'model', 'history.json')
    save_history(history_path, history)

    # Save model
    model_path = os.path.join(opt.results_dir, 'model', 'model.h5')
    save_model(model_path, model)


if __name__ == '__main__':
    opt = TrainOptions().parse()

    train(opt)
