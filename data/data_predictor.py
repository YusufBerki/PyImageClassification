import os

import numpy as np
import pandas as pd


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
        self.y_pred_classes = self.model.predict(self.data_generator)
        self.y_pred_classes = np.argmax(self.y_pred_classes, axis=1)

        self.label_map = self.data_generator.class_indices
        self.reversed_label_map = {v: k for k, v in self.data_generator.class_indices.items()}

        self.save_prediction_results()

    def save_prediction_results(self):
        prediction_results = []
        for index, value in enumerate(self.y_pred_classes):
            image_name = self.data_generator.filenames[index]
            prediction_results.append({
                "image": image_name,
                "real": self.reversed_label_map[self.y_test[index]],
                "prediction": self.reversed_label_map[value]
            })
        df = pd.DataFrame(prediction_results)
        df.to_excel(os.path.join(self.opt.results_dir, 'prediction_results.xlsx'), index=False)
