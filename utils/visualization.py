import os
import seaborn as sns
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from utils.utils import makedir


class Visualization:
    def __init__(self, opt, history, predictor=None):
        # Params
        self.history = history
        self.predictor = predictor

        # Directories
        self.results_dir = opt.results_dir
        self.charts_dir = makedir(os.path.join(opt.results_dir, 'charts'))

        # Check data generator
        if self.predictor:
            # Confusion Matrix
            self.get_confusion_matrix()

            # ROC Curve
            self.get_roc_curve()

        # Accuracy Chart
        self.get_accuracy_chart()

        # Loss Chart
        self.get_loss_chart()

    def get_confusion_matrix(self):
        cm = confusion_matrix(self.predictor.y_test, self.predictor.y_pred_classes)
        plt.figure(figsize=(6, 5))
        sns_plot = sns.heatmap(cm, fmt='n', annot=True, cmap='Blues')
        sns_plot.tick_params(labeltop=True, labelbottom=False, top=True, bottom=False)
        sns_plot.set_xlabel('Prediction')
        sns_plot.xaxis.set_label_position('top')
        sns_plot.set_ylabel('Real')

        sns_plot.set_xticklabels(self.predictor.label_names)
        sns_plot.set_yticklabels(self.predictor.label_names)

        sns_plot.figure.savefig(os.path.join(self.charts_dir, 'confusion_matrix.jpg'))

    def get_accuracy_chart(self):
        plt.figure()
        plt.plot(self.history['accuracy'])
        plt.legend(['Train'])
        plt.title('Model Accuracy (%)')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.savefig(os.path.join(self.charts_dir, 'accuracy.jpg'))

    def get_loss_chart(self):
        plt.figure()
        plt.plot(self.history['loss'])
        plt.legend(['Loss'])
        plt.title('Model Loss (%)')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.savefig(os.path.join(self.charts_dir, 'loss.jpg'))

    def get_roc_curve(self):
        y_test = self.predictor.y_test.copy()
        y_test = [self.predictor.label_names[index] for index in y_test]

        skplt.metrics.plot_roc(y_test,
                               self.predictor.y_pred,
                               plot_micro=False,
                               plot_macro=False,
                               )
        plt.title('ROC Curve')
        plt.xlabel('1 - Specificity')
        plt.ylabel('Sensitivity')
        plt.savefig(os.path.join(self.charts_dir, 'roc_curve.jpg'))
