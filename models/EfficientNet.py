from efficientnet.tfkeras import EfficientNetB5
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


def efficient_net_b5(opt):
    model = Sequential()
    enetb5 = EfficientNetB5(
        weights=opt.weights,
        include_top=opt.include_top,
        input_shape=(opt.input_shape_x, opt.input_shape_y, opt.input_shape_channel)
    )
    model.add(enetb5)
    model.add(layers.GlobalAvgPool2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(opt.number_of_classes, activation=opt.activation))
    model.compile(
        loss=opt.loss,
        optimizer=opt.optimizer,
        metrics=opt.metrics
    )
    return model
