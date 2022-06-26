import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_save_loss_acc(
    model_name: str, epoch: int, data: tuple, path_to_save_plot: str = None
) -> None:

    fig = plt.figure(figsize=(15, 5))

    ax = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)

    axis_labels = ["loss", "acc"]
    titles = ["Training/Validation loss.", "Training/Validation accuracy."]

    train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist = data

    # Train loss history plot
    ax.plot(
        np.arange(1, len(train_loss_hist) + 1),
        train_loss_hist,
        c="b",
        label="Training loss",
    )
    # Val loss history plot
    ax.plot(
        np.arange(1, len(val_loss_hist) + 1),
        val_loss_hist,
        c="g",
        label="Validation loss",
    )

    # Train loss accuracy plot
    ax1.plot(
        np.arange(1, len(train_acc_hist) + 1),
        train_acc_hist,
        c="b",
        label="Training accuracy",
    )

    # Val loss accuracy plot
    ax1.plot(
        np.arange(1, len(val_acc_hist) + 1),
        val_acc_hist,
        c="g",
        label="Validation accuracy",
    )

    for ax in fig.axes:
        axis_index = fig.axes.index(ax)
        axis_label = axis_labels[axis_index]
        ax.set_xlabel("epochs")
        ax.set_ylabel(axis_label)
        ax.set_title(titles[axis_index])
        ax.legend()

    plt.show()

    if path_to_save_plot and os.path.exists(path_to_save_plot):
        file_name = f"{int(time.time())}_{model_name}_epoch_{epoch}_plot.png"
        fig.savefig(path_to_save_plot + file_name)
    else:
        raise RuntimeError(f'Folder "./plots" does not exist in project structure.')


def plot_confusion_matrix(
    confusion_matrix, model_name: str, path_to_save_plot: str = None
):
    fig = plt.figure(figsize=(15, 10))

    class_labels = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

    df_cm = pd.DataFrame(
        confusion_matrix.astype(int), index=class_labels, columns=class_labels
    )
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=0, ha="right", fontsize=15
    )

    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=45, ha="right", fontsize=15
    )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

    if path_to_save_plot and os.path.exists(path_to_save_plot):
        fname = f"{int(time.time())}_cm_{model_name}_test.png"
        fig.savefig(path_to_save_plot + fname)
    else:
        raise RuntimeError(f'Folder "./plots" does not exist in project structure.')


def plot_metrics(
    metrics: dict, path_to_save_plot: str, model_name: str, metric_type: str
):
    fig = plt.figure(figsize=(15, 10))
    if metric_type not in metrics.keys():
        raise RuntimeError("Inaproperiate metric type.")
    if metric_type == "avg":
        metrics = metrics.get(metric_type)
        data = {
<<<<<<< HEAD
            "metrics": ["Accuracy", "Precision", "Recall", "F1 score"],
=======
            "metrics": ["Accuracy", "Precision", "Recall", "F1 score",],
>>>>>>> master
            "scores": [metrics[-2], metrics[1], metrics[0], metrics[-1]],
        }
        sns.barplot(x="metrics", y="scores", data=data)
        plt.title("Average metrics for all classes")
        plt.ylabel("Scores")
        plt.xlabel("Metrics")
    elif metric_type == "per_class":
        metrics_scores = np.vstack(metrics.get(metric_type))
        metrics = ["Accuracy", "Precision", "Recall", "F1 score"]
        class_labels = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
        scores = []
        for l_index in range(len(class_labels)):
            for m_index in range(len(metrics)):
                scores.append(
                    [
                        class_labels[l_index],
                        metrics[m_index],
                        metrics_scores[m_index, l_index],
                    ]
                )
        df = pd.DataFrame(scores, columns=["Mole Type", "Metric", "Score"])
        sns.barplot(data=df, x="Mole Type", y="Score", hue="Metric")

    plt.show()
    if path_to_save_plot and os.path.exists(path_to_save_plot):
        fname = f"{int(time.time())}_metrics_{model_name}_test_{metric_type}.png"
        fig.savefig(path_to_save_plot + fname)
    else:
        raise RuntimeError(f'Folder "./plots" does not exist in project structure.')
