# experiments/metrics.py
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import os

def compute_cm(true_labels, preds, class_names, out_path):
    cm = confusion_matrix(true_labels, preds)
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center', color='white' if cm[i,j]>cm.max()/2 else 'black')
    plt.ylabel("True")
    plt.xlabel("Predicted")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return cm

def print_classification_report(true_labels, preds, class_names):
    print(classification_report(true_labels, preds, target_names=class_names))
