import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, auc
)
import json

class ModelEvaluator:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
    
    def compute_metrics(self):
        """
        Compute classification metrics
        """
        accuracy = accuracy_score(self.y_true, self.y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_true, self.y_pred, average='binary'
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # Save metrics to JSON
        with open('data/processed/evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f)
        
        return metrics
    
    def plot_confusion_matrix(self):
        """
        Generate confusion matrix visualization
        """
        cm = confusion_matrix(self.y_true, self.y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Skin Cancer Classification Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig('visualizations/confusion_matrix.png')
        plt.close()
    
    def plot_roc_curve(self, y_scores):
        """
        Generate ROC curve
        """
        fpr, tpr, _ = roc_curve(self.y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig('visualizations/roc_curve.png')
        plt.close()