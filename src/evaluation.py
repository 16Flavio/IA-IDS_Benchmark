import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import numpy as np
import os

class Evaluator:
    """
    Handles evaluation metric calculation and visualization for model performance.
    """
    def __init__(self, output_dir='results'):
        """
        Initialize the Evaluator.

        Args:
            output_dir (str): Directory where evaluation plots and reports will be saved.
        """
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """
        Plot and save the confusion matrix.

        Args:
            y_true (array-like): Ground truth (correct) target values.
            y_pred (array-like): Estimated targets as returned by a classifier.
            model_name (str): Name of the model for the plot title and filename.
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Matrice de Confusion - {model_name}')
        plt.xlabel('Prédiction')
        plt.ylabel('Réalité')
        
        filename = f"{model_name.replace(' ', '_')}_cm.png"
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        print(f"   [Graphique] Matrice de confusion sauvegardée : {filename}")

    def plot_roc_curve(self, y_true, y_probs, model_name):
        """
        Plot and save the Receiver Operating Characteristic (ROC) curve.

        Args:
            y_true (array-like): True binary labels.
            y_probs (array-like): Target scores, can either be probability estimates of the positive class or confidence values.
            model_name (str): Name of the model.
        """
        if y_probs is None:
            return
            
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taux de Faux Positifs')
        plt.ylabel('Taux de Vrais Positifs')
        plt.title(f'Courbe ROC - {model_name}')
        plt.legend(loc="lower right")
        
        filename = f"{model_name.replace(' ', '_')}_roc.png"
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        print(f"   [Graphique] Courbe ROC sauvegardée : {filename}")

    def save_report(self, y_true, y_pred, model_name):
        """
        Generate and save a text classification report.

        Args:
            y_true (array-like): Ground truth values.
            y_pred (array-like): Estimated targets.
            model_name (str): Name of the model.
        """
        report = classification_report(y_true, y_pred)
        filename = f"{model_name.replace(' ', '_')}_report.txt"
        with open(os.path.join(self.output_dir, filename), "w") as f:
            f.write(report)