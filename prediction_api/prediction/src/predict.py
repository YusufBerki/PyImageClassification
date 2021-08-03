import os
from prediction.src.opt import model_path, classes
from prediction.src.utils import load_model, load_image

if os.path.exists(model_path):
    model = load_model(model_path)
    print("Model successfully loaded!")
else:
    print("Model file path does not exist or is incorrect! Please check prediction/src/opt.py")


def predict_image(image_path):
    image = load_image(image_path)
    results = model.predict(image)
    results = [{'name': classes[index], 'proba': float(proba)} for index, proba in enumerate(list(results[0]))]
    return results
