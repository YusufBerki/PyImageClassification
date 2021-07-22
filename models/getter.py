from models.EfficientNet import efficient_net_b5
from opt import MODEL_PARAMS

models = {
    "efficient_net_b5": efficient_net_b5
}


def model_getter():
    return models[MODEL_PARAMS['model']]()
