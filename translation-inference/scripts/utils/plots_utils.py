import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import metrics
import os
from typing import List


def plot_histogram(data: dict | list, output_path: str, x_label: str = '', y_label: str = '', title: str = '', legend: List[str] = None):
    """
    Plots a histogram from a dictionary.

    Args:
        data (dict or list): A dictionary where keys represent categories and values represent corresponding values,
                             or a list of such dictionaries.
        output_path (str): The path to save the figure.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        title (str): Title for the plot.
        legend (list of strings): Legend for the plot.
    """

    # Check if data is a single dictionary or a list of dictionaries
    if isinstance(data, dict):
        data_dicts = [data]
    else:
        data_dicts = data

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    # Width of each bar
    bar_width = 0.8 / len(data_dicts)
    # List to store bar containers
    bar_containers = []

    # Iterate over the dictionaries
    for i, data_dict in enumerate(data_dicts):
        categories = list(data_dict.keys())
        values = list(data_dict.values())
        # Shift the bars horizontally to display them side by side
        bar_positions = [x + i * bar_width for x in range(len(categories))]
        # Plot the bars
        bars = ax.bar(bar_positions, values, width=bar_width, label=legend[i] if legend else None)
        bar_containers.append(bars)
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                # height,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    # Set x-axis, y-axis and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=45, ha='right')
    # Set x-axis tick labels
    ax.set_xticks([x + bar_width * (len(data_dicts) - 1) / 2 for x in range(len(categories))])
    ax.set_xticklabels(categories)
    # Add legend if provided
    if legend:
        ax.legend(bar_containers, legend)

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path)


def plot_delta_hist(save_path: str, metric_name: str, delta_arr: np.array, delta_mean: float, positive_rate: float = None):
    plt.figure(figsize=(10, 6))
    counts, bins, _ = plt.hist(delta_arr, bins=80, density=False)
    plt.axvline(delta_mean, color='g', label='delta mean', ls='dashed')
    if positive_rate is None:
        plt.title(f"{metric_name} delta")
    else:
        plt.title(f"{metric_name} delta, positive rate: {round(positive_rate, 2)}%, mean = {round(delta_mean, 2)}")
    plt.ylabel("Counts")
    plt.xlabel("delta")
    plt.legend()
    plt.savefig(save_path + f"/delta_hist_{metric_name}.png")
    plt.clf()


def plot_diff_hist(compare_pairs: list[tuple] | tuple, labels: list[str] | str, save_path: str, subtraction_restrictions=None):
    """
    Plots histograms for the differences between pairs of data.

    Parameters:
        compare_pairs (list of tuples or a single tuple): Can be a list of tuples where each tuple
                                                          contains two arrays to compare, or a single tuple.
        labels (list of str or a single str): The metrics names we use to compare each pair.
        save_path (str): Path to save the histogram plots.
        subtraction_restrictions (callable, optional): A custom function for calculating the differences
                                                       between pairs. If None, regular subtraction is used.

    Returns:
        None
    """
    if not isinstance(compare_pairs, list):
        compare_pairs = [compare_pairs]
    if not isinstance(labels, list):
        labels = [labels]
    if len(compare_pairs) != len(labels):
        raise ValueError("There should be a label for each pair")
    for i, pair in enumerate(compare_pairs):
        if len(pair) != 2:
            raise ValueError(f"Each pair must contain exactly 2 candidates. pair #{i} doesn't match this requirement")
        print(labels[i])
        cand1 = np.array(pair[0])
        cand2 = np.array(pair[1])

        # Use the custom subtraction function if provided, otherwise use regular subtraction
        if subtraction_restrictions:
            print(f"Using '{subtraction_restrictions.__name__}' function for subtraction")
            delta = subtraction_restrictions(cand2, cand1)
        else:
            delta = cand2 - cand1

        mean_delta = np.mean(delta)
        positive = np.where(delta >= 0)[0]
        positive_rate = positive.shape[0] / delta.shape[0]
        # plot delta hist
        plot_delta_hist(save_path, labels[i], delta, mean_delta, positive_rate=positive_rate)


def cm(gts, preds, model_name, labels, vis_res):
    confusion_matrix = metrics.confusion_matrix(preds, gts)
    display_labels = sorted(list(set(gts).union(set(preds))))
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=display_labels)
    fig, ax = plt.subplots(figsize=(10, 10))  # You can specify the size of the figure
    cm_display.plot(ax=ax, cmap=plt.cm.Blues)

    # Set custom axis titles
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])

    acc = metrics.accuracy_score(preds, gts)
    kappa = metrics.cohen_kappa_score(preds, gts, weights='quadratic')
    plt.title(f'{model_name} \nAccuracy: {acc}, Weighted Kappa: {kappa}')

    # Save the figure
    path = os.path.join(vis_res, f'confusion_matrix_{model_name}.png')
    plt.savefig(path)


def plot_corrs(df, path, type, method="pearson"):
    """
    plot correlation matrix
    """
    corr_df = df.corr(method)
    plt.figure(figsize=(18, 18))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 20})
    plt.title(f'CORR {type}')
    corr_png_name = f'{method}_{type}_CORRS.png'
    plt.savefig(os.path.join(path, corr_png_name))


def roc_optimal(scores, labels, model_name, vis_res):

    # Calculate ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)

    # Find the optimal threshold ny maximizing the diff between TPR and FPR
    diffs = tpr - fpr
    optimal_idx = np.argmax(diffs)
    optimal_threshold = thresholds[optimal_idx]
    half_threshold_index = np.argmin(np.abs(thresholds - 0.5))

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, marker='.')
    plt.plot(fpr[half_threshold_index], tpr[half_threshold_index], 'yo', label='Threshold=0.5')
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', label='Optimal Threshold')  # Highlight the optimal point
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower left')

    # Save the figure
    path = os.path.join(vis_res, f'ROC_Curve_{model_name}.png')
    plt.savefig(path)

    # Print and return the optimal threshold
    print("ROC Optimal Threshold:", optimal_threshold)
    print("TPR-FPR at Optimal Threshold:", diffs[optimal_idx])
    print("AUC:", metrics.auc(fpr, tpr))


def pr_optimal(scores, labels, model_name, vis_res):
    # Precision-Recall curve
    precision, recall, pr_thresholds = metrics.precision_recall_curve(labels, scores)
    pr_auc = metrics.auc(recall, precision)

    # Calculate F1 score for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall)

    # Find the thresholds that maximize F1 score, recall, and precision
    optimal_f1_idx = np.argmax(f1_scores)
    optimal_recall_idx = np.argmax(recall)
    optimal_precision_idx = np.argmax(precision[:-1])  # The last precision is always 1.0, we ignore it
    half_threshold_index = np.argmin(np.abs(pr_thresholds - 0.5))

    optimal_f1_threshold = pr_thresholds[optimal_f1_idx]
    optimal_recall_threshold = pr_thresholds[optimal_recall_idx]
    optimal_precision_threshold = pr_thresholds[optimal_precision_idx]

    # Plot Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label='PR curve')

    # Highlight the optimal points
    plt.plot(recall[half_threshold_index], precision[half_threshold_index], 'yo', label='Threshold=0.5')
    plt.plot(recall[optimal_f1_idx], precision[optimal_f1_idx], 'ro', label='Optimal F1 Threshold')
    plt.plot(recall[optimal_recall_idx], precision[optimal_recall_idx], 'go', label='Optimal Recall Threshold')
    plt.plot(recall[optimal_precision_idx], precision[optimal_precision_idx], 'bo', label='Optimal Precision Threshold')

    # Add labels and title
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve ({model_name})')
    plt.legend(loc='lower left')

    # Save the figure
    path = os.path.join(vis_res, f'PR_Curve_{model_name}.png')
    plt.savefig(path)

    # Print and return the optimal thresholds
    print("Optimal Thresholds:")
    print(f"F1 Optimal Threshold: {optimal_f1_threshold}")
    print(f"Values at F1 Optimal Threshold: recall={recall[optimal_f1_idx]}, precision={precision[optimal_f1_idx]}")
    print(f"Recall Optimal Threshold: {optimal_recall_threshold}")
    print(f"Values at Recall Optimal Threshold: recall={recall[optimal_recall_idx]}, precision={precision[optimal_recall_idx]}")
    print(f"Precision Optimal Threshold: {optimal_precision_threshold}")
    print(f"Values at Precision Optimal Threshold: recall={recall[optimal_precision_idx]}, precision={precision[optimal_precision_idx]}")
    print("AUC:", pr_auc)


def plot_accumulated_percentage(df: pd.DataFrame, save_path: str):
    """
    Plots the accumulated percentage of data values in the range 1-5 and the mean value.

    Parameters:
    df (Dataframe): each column is a list of values in the range 1-5.
    """

    plt.figure(figsize=(12, 8))

    colors = ['skyblue', 'lightgreen']
    mean_lines = ['r--', 'b--']

    columns = df.columns
    for i, column in enumerate(columns):
        # Extract the data from the DataFrame
        data = df[column].values

        # Calculate accumulated percentage
        value_counts = np.bincount(data, minlength=6)[1:]  # Count occurrences of each value (ignore zero index)
        accumulated_counts = np.cumsum(value_counts)
        total_counts = len(data)
        accumulated_percentage = (accumulated_counts / total_counts) * 100

        # Calculate mean value
        mean_value = np.mean(data)

        # Plotting
        bars = plt.bar(np.arange(1, 6) + i * 0.3, accumulated_percentage, width=0.3, color=colors[i], edgecolor='black',
                       label=f'{column} Accumulated Percentage')

        # Adding mean value line
        plt.axvline(x=mean_value, color=mean_lines[i][0], linestyle='--',
                    label=f'{column} Mean Value: {mean_value:.2f}')

        # Annotate bars with the percentage values
        for bar, percentage in zip(bars, accumulated_percentage):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 5, f'{percentage:.1f}%', ha='center',
                     va='bottom')

    plt.legend()
    # Adding titles and labels
    plt.title('Accumulated Percentage and Mean Value')
    plt.xlabel('Value')
    plt.ylabel('Accumulated Percentage (%)')
    plt.xticks(range(1, 6))

    # Save the figure
    plt.savefig(save_path)

