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
    history_data = pandas.DataFrame(
        {"Accuracy": history.history["accuracy"],
         "Val Accuracy": history.history["val_accuracy"],
         "Loss": history.history["loss"],
         "Val Loss": history.history["val_loss"]})
    file_path = Path(filename)
    history_data.to_csv(file_path)
    return file_path.absolute()


def draw_conf_mat(matrix, filename):
    labels = ['class 0', 'class 1']
    ax = plt.subplot()
    sns.heatmap(matrix, annot=True, ax=ax)
    plt.title('Confusion matrix')
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(filename)


if __name__ == "__main__":

    filename = sys.argv[1]
