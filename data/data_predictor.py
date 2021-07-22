class Predictor:
    def __init__(self, opt, model, data_generator):
        # Params
        self.opt = opt
        self.model = model
        self.data_generator = data_generator

        # Real classes
        self.y_test = self.data_generator.classes

        # Predictions
        self.y_pred = self.model.predict_generator(self.data_generator)

        # Prediction Classes
        self.y_pred_classes = self.model.predict_classes(self.data_generator)
