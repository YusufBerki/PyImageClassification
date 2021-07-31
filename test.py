from options.test_options import TestOptions

from utils.utils import load_history, load_model

from data.data_generator import get_data_generator
from data.data_loader import get_data_loader
from data.data_predictor import Predictor

from utils.visualization import Visualization

import os
import warnings

warnings.filterwarnings("ignore")


def test(opt):
    # Get data generator
    data_generator = get_data_generator(opt)

    # Load data
    data_generator = get_data_loader(opt, data_generator, opt.test_dataset_dir)

    # Load history
    history_path = os.path.join(opt.results_dir, 'model', 'history.json')
    history = load_history(history_path)

    # Load model
    model_path = os.path.join(opt.results_dir, 'model', 'model.h5')
    model = load_model(model_path)

    # Predict Data
    predictor = Predictor(opt, model, data_generator)

    # Visualization
    visualization = Visualization(opt, history, predictor)


if __name__ == '__main__':
    opt = TestOptions().parse()
    test(opt)
