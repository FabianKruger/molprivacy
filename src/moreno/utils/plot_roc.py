from typing import List
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def plot_roc(y_pred: List[float], y_true: List[float], save_location: Path):
    """
    Plot ROC curve using predicted probabilities and true labels.

    Parameters:
    y_pred: List[float] - The predicted probabilities of the positive class.
    y_true: List[float] - The true labels (binary).
    """

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # Set the plot style using seaborn
    sns.set(style="whitegrid")

    # Create the ROC plot
    plt.figure(figsize=(8, 6))
    sns.lineplot(
        x=fpr, y=tpr, label=f"ROC curve (area = {roc_auc:.2f})", color="blue", lw=2
    )

    # Plot the diagonal (random performance) line
    sns.lineplot(x=[0, 1], y=[0, 1], linestyle="--", color="gray", lw=2)

    # Customize the plot with labels, title, etc.
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Model Performance", fontsize=14)

    # Add a legend
    plt.legend(loc="lower right")

    # Save the plot as a PDF at the specified location
    plt.savefig(save_location / "model_performance.pdf", format="pdf")

    # Close the plot to free up memory
    plt.close()
