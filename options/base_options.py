import argparse
import os
from utils.utils import fix_opt_paths, get_now, makedirs


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        self.initialized = True
        parser.add_argument('--model', type=str, default='InceptionV3')
        parser.add_argument('--dataset_path', type=str, default='dataset')
        parser.add_argument('--train_dataset_dir', type=str, default='train')
        parser.add_argument('--test_dataset_dir', type=str, default='test')
        parser.add_argument('--results_dir', type=str, default='results')

        # Data Loader
        parser.add_argument('--target_size_x', type=int, default=150)
        parser.add_argument('--target_size_y', type=int, default=150)
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--shuffle', type=bool, default=True)
        parser.add_argument('--seed', type=int, default=32)
        parser.add_argument('--interpolation', type=str, default='nearest')

        parser.add_argument('--gpu_ids', type=str, default=None)

        return parser

    def parse(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser
        self.opt = self.parser.parse_args()

        self.opt = fix_opt_paths(self.opt)

        self.opt.is_train = self.is_train
        if self.opt.is_train:
            self.opt.results_dir = makedirs(os.path.join(self.opt.results_dir, f'{self.opt.model}_{get_now()}'))
        self.opt.number_of_classes = len(os.listdir(self.opt.train_dataset_dir))

        self.print_options()

        return self.opt

    def print_options(self):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(self.opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:  # check if it is not default
                comment = f'\t[default: {default}]'
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)