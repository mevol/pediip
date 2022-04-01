"""Plot the history of a keras model"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def figure_from_csv(history_file, filename):
    history = pd.read_csv(history_file)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    f.suptitle(Path(history_file).stem, fontsize=12)
    f.subplots_adjust(top=0.85, wspace=0.3)
    epochs = len(history["Accuracy"])
    epoch_list = range(epochs)
    ax1.plot(epoch_list, history["Accuracy"], label="Train Accuracy")
    ax1.plot(epoch_list, history["Val Accuracy"], label="Validation Accuracy")
    ax1.set_ylabel("Accuracy Value")
    ax1.set_xlabel("Epoch")
    ax1.set_title("Accuracy")
    ax1.legend(loc="best")
    ax2.plot(epoch_list, history["Loss"], label="Train Loss")
    ax2.plot(epoch_list, history["Val Loss"], label="Validation Loss")
    ax2.set_ylabel("Loss Value")
    ax2.set_xlabel("Epoch")
    ax2.set_title("Loss")
    ax2.legend(loc="best")
    plt.savefig(filename)
    return f


def history_to_csv(history, filename):
    """Put the history in a csv file with known format"""
    history_data = pd.DataFrame(
        {"Accuracy": history.history["accuracy"],
         "Val Accuracy": history.history["val_accuracy"],
         "Loss": history.history["loss"],
         "Val Loss": history.history["val_loss"]})
    file_path = Path(filename)
    history_data.to_csv(file_path)
    return file_path.absolute()


def confusion_matrix_and_stats(y_test, y_pred, filename):
    # Plot predictions in confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred)

    # draw confusion matrix
    labels = ['0', '1']      
    ax = plt.subplot()
    sns.heatmap(conf_mat, annot=True, ax=ax)
    plt.title('Confusion matrix of the classifier')
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(filename, dpi=600)
    plt.close()

    # separating prediction outcomes in TP, TN, FP, FN
    TP = conf_mat[1, 1]
    TN = conf_mat[0, 0]
    FP = conf_mat[0, 1]
    FN = conf_mat[1, 0]
    
    acc_score = round(((TP + TN) / (TP + TN + FP + FN)) * 100, 2)
    class_err = round(((FP + FN) / (TP + TN + FP + FN)) * 100, 2)
    sensitivity = round((TP / (FN + TP)) * 100, 2)
    specificity = round((TN / (TN + FP)) * 100, 2)
    false_positive_rate = round((FP / (TN + FP)) * 100, 2)
    false_negative_rate = round((FN / (TP + FN)) * 100, 2)
    precision = round((TP / (TP + FP)) * 100, 2)
    f1 = round(f1_score(y_test, y_pred) * 100, 2)

    conf_mat_dict = {'TP' : TP,
                     'TN' : TN,
                     'FP' : FP,
                     'FN' : FN,
                     'acc' : acc_score,
                     'err' : class_err,
                     'sensitivity' : sensitivity,
                     'specificity' : specificity,
                     'FP-rate' : false_positive_rate,
                     'FN-rate' : false_negative_rate,
                     'precision' : precision,
                     'F1-score' : f1}
    return conf_mat_dict


#plot precision and recall curve
def plot_precision_recall_vs_threshold(y_test, y_pred_proba_ones, filename):
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba_ones)
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="Recall")
    plt.title('Precsion-Recall plot test set class 1')
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0,1])
    plt.savefig(filename, dpi=600)
    plt.close()


#plot ROC curves
def plot_roc_curve(y_test, y_pred_proba_ones, filename):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_ones)
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.title('ROC curve for classifier on test set for class 1') 
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.grid(True)
    plt.savefig(filename, dpi=600)
    plt.close()
    return fpr, tpr, thresholds


if __name__ == "__main__":

    filename = sys.argv[1]
