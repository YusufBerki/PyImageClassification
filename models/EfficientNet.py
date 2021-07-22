from efficientnet.tfkeras import EfficientNetB5
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from opt import MODEL_PARAMS


def efficient_net_b5():
    model = Sequential()
    enetb5 = EfficientNetB5(**MODEL_PARAMS['params'])
    model.add(enetb5)
    model.add(layers.GlobalAvgPool2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(MODEL_PARAMS['number_of_classes'], activation=MODEL_PARAMS['activation']))
    model.compile(loss=MODEL_PARAMS['loss'], optimizer=MODEL_PARAMS['optimizer'], metrics=MODEL_PARAMS['metrics'])
    return model
