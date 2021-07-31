from tensorflow.keras import applications
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model

models = {
    # DenseNet
    "DenseNet121": applications.DenseNet121,
    "DenseNet169": applications.DenseNet169,
    "DenseNet201": applications.DenseNet201,

    # EfficientNet
    "EfficientNetB0": applications.EfficientNetB0,
    "EfficientNetB1": applications.EfficientNetB1,
    "EfficientNetB2": applications.EfficientNetB2,
    "EfficientNetB3": applications.EfficientNetB3,
    "EfficientNetB4": applications.EfficientNetB4,
    "EfficientNetB5": applications.EfficientNetB5,
    "EfficientNetB6": applications.EfficientNetB6,
    "EfficientNetB7": applications.EfficientNetB7,

    # Inception
    "InceptionV3": applications.InceptionV3,
    "InceptionResNetV2": applications.InceptionResNetV2,

    # MobileNet
    "MobileNet": applications.MobileNet,
    "MobileNetV2": applications.MobileNetV2,
    "MobileNetV3Large": applications.MobileNetV3Large,
    "MobileNetV3Small": applications.MobileNetV3Small,

    # NasNet
    "NASNetLarge": applications.NASNetLarge,
    "NASNetMobile": applications.NASNetMobile,

    # ResNet
    "ResNet101": applications.ResNet101,
    "ResNet152": applications.ResNet152,
    "ResNet50": applications.ResNet50,
    "ResNet101V2": applications.ResNet101V2,
    "ResNet152V2": applications.ResNet152V2,
    "ResNet50V2": applications.ResNet50V2,

    # VGG
    "VGG16": applications.VGG16,
    "VGG19": applications.VGG19,

    # Xception
    "Xception": applications.Xception,
}


def model_getter(opt):
    # Get transfer learning model
    model = models[opt.model](
        weights=opt.weights,
        include_top=opt.include_top,
        input_shape=(opt.input_shape_x, opt.input_shape_y, opt.input_shape_channel)
    )

    # Set trainable attribute to False
    for layer in model.layers:
        layer.trainable = False

    # Set model output to x
    x = model.output

    # Flatten layer
    x = Flatten()(x)

    # Number of fully connected layer loop
    for _ in range(opt.number_of_fc_layer):
        # New fully connected layer
        x = Dense(opt.fc_layer_size, activation=opt.fc_activation)(x)

        # Dropout
        x = Dropout(opt.dropout)(x)

    # New softmax layer
    outputs = Dense(opt.number_of_classes, activation=opt.activation)(x)

    # Create finetune model
    model = Model(inputs=model.input, outputs=outputs)

    # Compile model for training/testing
    model.compile(
        loss=opt.loss,
        optimizer=opt.optimizer,
        metrics=opt.metrics
    )
    return model
