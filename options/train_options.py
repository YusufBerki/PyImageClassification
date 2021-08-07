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
        parser.add_argument('--brightness_min', type=float, default=None)
        parser.add_argument('--brightness_max', type=float, default=None)
        parser.add_argument('--shear_range', type=float, default=0.)
        parser.add_argument('--zoom_min', type=float, default=0.)
        parser.add_argument('--zoom_max', type=float, default=0.)
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
        parser.add_argument('--model_checkpoint_filepath', type=str, default='epoch_{epoch:02d}_loss_{loss:.2f}.hdf5')
        parser.add_argument('--model_checkpoint_save_weights_only', type=bool, default=True)
        parser.add_argument('--model_checkpoint_monitor', type=str, default='accuracy')
        parser.add_argument('--model_checkpoint_mode', type=str, default='max')
        parser.add_argument('--model_checkpoint_save_best_only', type=bool, default=True)
        parser.add_argument('--model_checkpoint_verbose', type=int, default=0)
        parser.add_argument('--model_checkpoint_save_freq', type=str, default="epoch")

        # Callbacks - Early Stopping
        parser.add_argument('--early_stopping', type=bool, default=False)
        parser.add_argument('--early_stopping_monitor', type=str, default="val_loss")
        parser.add_argument('--early_stopping_min_delta', type=float, default=0)
        parser.add_argument('--early_stopping_patience', type=float, default=0)
        parser.add_argument('--early_stopping_verbose', type=float, default=0)
        parser.add_argument('--early_stopping_mode', type=str, default='auto')
        parser.add_argument('--early_stopping_baseline', type=float, default=False)
        parser.add_argument('--early_stopping_restore_best_weights', type=bool, default=False)

        # Callbacks - TensorBoard
        parser.add_argument('--tensorboard', type=bool, default=False)
        parser.add_argument('--tensorboard_log_dir', type=str, default='tensorboard_logs')
        parser.add_argument('--tensorboard_histogram_freq', type=int, default=0)
        parser.add_argument('--tensorboard_write_graph', type=bool, default=True)
        parser.add_argument('--tensorboard_write_images', type=bool, default=False)
        parser.add_argument('--tensorboard_write_steps_per_second', type=bool, default=False)
        parser.add_argument('--tensorboard_update_freq', type=str, default="epoch")
        parser.add_argument('--tensorboard_profile_batch', type=int, default=2)
        parser.add_argument('--tensorboard_embeddings_freq', type=int, default=0)

        # Training
        parser.add_argument('--resume', type=bool, default=False)
        parser.add_argument('--weights', type=str, default='imagenet')
        parser.add_argument('--include_top', type=bool, default=False)
        parser.add_argument('--input_shape_x', type=int, default=150)
        parser.add_argument('--input_shape_y', type=int, default=150)
        parser.add_argument('--input_shape_channel', type=int, default=3)
        parser.add_argument('--number_of_fc_layer', type=int, default=2)
        parser.add_argument('--fc_layer_size', type=int, default=1024)
        parser.add_argument('--fc_activation', type=str, default='relu')
        parser.add_argument('--dropout', type=float, default=0.5)
        parser.add_argument('--activation', type=str, default='softmax')
        parser.add_argument('--loss', type=str, default='categorical_crossentropy')
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--metrics', type=str, default=["accuracy"])
        parser.add_argument('--epoch', type=int, default=10)

        self.is_train = True
        return parser
