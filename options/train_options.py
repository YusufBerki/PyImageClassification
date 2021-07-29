from options.base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # Data Generator
        parser.add_argument('--horizontal_flip', type=bool, default=False)
        parser.add_argument('--vertical_flip', type=bool, default=False)
        parser.add_argument('--featurewise_center', type=bool, default=False)
        parser.add_argument('--samplewise_center', type=bool, default=False)
        parser.add_argument('--featurewise_std_normalization', type=bool, default=False)
        parser.add_argument('--samplewise_std_normalization', type=bool, default=False)
        parser.add_argument('--rotation_range', type=float, default=0)
        parser.add_argument('--width_shift_range', type=float, default=0.)
        parser.add_argument('--height_shift_range', type=float, default=0.)
        parser.add_argument('--brightness_range', type=list, default=None)
        parser.add_argument('--shear_range', type=float, default=0.)
        parser.add_argument('--zoom_range', type=float, default=0.)
        parser.add_argument('--channel_shift_range', type=float, default=0.)
        parser.add_argument('--fill_mode', type=str, default='nearest')

        # Callbacks
        # Callbacks - Reduce Learning Rate On Plateau
        parser.add_argument('--reduce_lr', type=bool, default=False)
        parser.add_argument('--reduce_lr_monitor', type=str, default='loss')
        parser.add_argument('--reduce_lr_factor', type=str, default=0.3)
        parser.add_argument('--reduce_lr_patience', type=str, default=5)
        parser.add_argument('--reduce_lr_verbose', type=str, default=2)
        parser.add_argument('--reduce_lr_mode', type=str, default='max')

        # Callbacks - Save Model Checkpoints
        parser.add_argument('--model_checkpoint', type=bool, default=False)
        parser.add_argument('--model_checkpoint_filepath', type=str, default=None)
        parser.add_argument('--model_checkpoint_save_weights_only', type=bool, default=True)
        
        # Callbacks - Early Stopping
        parser.add_argument('--early_stopping', type=bool, default=True)
        parser.add_argument('--early_stopping_monitor', type=str, default="val_loss")
        parser.add_argument('--early_stopping_min_delta', type=float, default=0)
        parser.add_argument('--early_stopping_patience', type=float, default=0)
        parser.add_argument('--early_stopping_verbose', type=float, default=0)
        parser.add_argument('--early_stopping_mode', type=str, default='auto')
        parser.add_argument('--early_stopping_baseline', type=float, default=False)
        parser.add_argument('--early_stopping_restore_best_weights', type=bool, default=False)

        # Training
        parser.add_argument('--weights', type=str, default='imagenet')
        parser.add_argument('--include_top', type=bool, default=False)
        parser.add_argument('--input_shape_x', type=int, default=150)
        parser.add_argument('--input_shape_y', type=int, default=150)
        parser.add_argument('--input_shape_channel', type=int, default=3)
        parser.add_argument('--activation', type=str, default='softmax')
        parser.add_argument('--loss', type=str, default='categorical_crossentropy')
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--metrics', type=str, default=["accuracy"])

        self.is_train = True
        return parser
