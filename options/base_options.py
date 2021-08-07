import argparse
import os
from utils.utils import fix_opt_paths, get_now, makedirs, makedir


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

        # Check train mode
        if self.opt.is_train:

            # Check resume mode
            if not self.opt.resume:
                # Create new training directory
                self.opt.results_dir = makedirs(os.path.join(self.opt.results_dir, f'{self.opt.model}_{get_now()}'))

            # Check save model checkpoint callbacks is active
            if self.opt.model_checkpoint:
                # Create checkpoints directory
                checkpoints_dir = makedir(os.path.join(self.opt.results_dir, 'checkpoints'))

                # Configure model checkpoints path
                self.opt.model_checkpoint_filepath = os.path.join(checkpoints_dir, self.opt.model_checkpoint_filepath)

        # Set number of classes
        self.opt.number_of_classes = len(os.listdir(self.opt.train_dataset_dir))

        # Print all options
        self.print_options()

        return self.opt

    def print_options(self):
        sep = '-' * 35

        message = ''
        message += f'{sep} Options {sep}\n'

        for k, v in sorted(vars(self.opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:  # check if it is not default
                comment = f'\t[default: {default}]'
            message += '{:>40}: {:<50}{}\n'.format(str(k), str(v), comment)

        message += f'{sep} End {sep}'
        print(message)
