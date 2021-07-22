from keras.preprocessing.image import ImageDataGenerator


def get_data_generator(opt):
    if opt.is_train:
        return ImageDataGenerator(
            featurewise_center=opt.featurewise_center,
            samplewise_center=opt.samplewise_center,
            featurewise_std_normalization=opt.featurewise_std_normalization,
            samplewise_std_normalization=opt.samplewise_std_normalization,
            rotation_range=opt.rotation_range,
            width_shift_range=opt.width_shift_range,
            height_shift_range=opt.height_shift_range,
            brightness_range=opt.brightness_range,
            shear_range=opt.shear_range,
            zoom_range=opt.zoom_range,
            channel_shift_range=opt.channel_shift_range,
            fill_mode=opt.fill_mode,
            horizontal_flip=opt.horizontal_flip,
            vertical_flip=opt.vertical_flip
        )
    return ImageDataGenerator()
