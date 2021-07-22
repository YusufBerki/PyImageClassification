from models.EfficientNet import efficient_net_b5

models = {
    "efficient_net_b5": efficient_net_b5
}


def model_getter(opt):
    return models[opt.model](opt)
