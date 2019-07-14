"""Plotting helper functions"""
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import OneHotEncoder

def plot_roc(y_true, y_score, num_classes, output_dir, fig_dir, name):
    """Save ROC plot and raw values.
    Arguments:
        y_true (array): Real labels
        Y_score (array): Model predictions
        num_classes (int): number of classes
        output_dir (string): Path to save output
        fig_dir (string): Path to save figures
        name (string): Name to save files under
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Onehot encode labels
    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    onehot_encoded = []
    for i in range(len(y_true)):
        onehot_encoded.append(onehot_encoder.fit_transform(
            y_true[i].reshape(len(y_true[i]), 1)))
    score = np.concatenate(y_score)
    true = np.concatenate(onehot_encoded)

    # Get false positive and tru positive for each class
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(true[:, i], score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(true.ravel(), score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Save output
    np.savetxt(f'{output_dir}{os.sep}{name}-roc-fpr.csv',
               fpr[1], fmt='%1.8f', delimiter=',', newline='\n')
    np.savetxt(f'{output_dir}{os.sep}{name}-roc-tpr.csv',
               tpr[1], fmt='%1.8f', delimiter=',', newline='\n')
    np.savetxt(f'{output_dir}{os.sep}{name}-roc-auc.csv',
               [roc_auc[1]], fmt='%1.8f', delimiter=',', newline='\n')

    # Save plot
    plt.figure()
    lw = 1
    plt.plot(fpr[1], tpr[1], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f'{fig_dir}{os.sep}{name}_roc.pdf')
    plt.clf()


def class_bar_plot(data_loader, fig_dir, name):
    """Save bar plot of class counts.
    Arguments:
        data_loader (object): pytorch dataloader
        fig_dir (string): Path to save figures
        name (string): Name to save files under
    """
    _, ax = plt.subplots()
    _, counts = np.unique(data_loader.dataset.targets, return_counts=True)
    ax.bar(data_loader.dataset.classes, counts)
    ax.set_xticks(data_loader.dataset.classes)
    ax.set_yticks(counts)
    plt.savefig(f'{fig_dir}{os.sep}{name}_class_counts.pdf')
    plt.clf()


def confusion_matrix_plot(data_loader, conf_mat, fig_dir, name):
    """Save confusion matrix plot
    Arguments:
        data_loader (object): pytorch dataloader
        conf_mat (object): sklearn confusion matrix
        fig_dir (string): Path to save figures
        name (string): Name to save files under
    """
    ticks = data_loader.dataset.classes
    sns.heatmap(conf_mat, annot=True, fmt='', cmap="Blues",
                xticklabels=ticks, yticklabels=ticks)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.yticks(rotation=0)
    plt.savefig(f'{fig_dir}{os.sep}{name}_conf_mat.pdf')
    plt.clf()
