import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from sklearn.metrics import auc


def load_auroc_values(dataset, job, representation):
    # implement based on your folder structure
    # example:
    # datapath = ../dataset/job/representation/model/
    # roc = pd.read_csv(datapath)
    # tpr = roc['tpr']
    # fpr = roc['fpr']
    # auroc = auc(fpr, tpr)
    # return auc
    pass


if __name__ == "__main__":
    representations = [
        "ECFP4",
        "ECFP6",
        "graph",
        "MACCS",
        "rdkit",
        "transformer_matrix",
    ]
    datasets = ["bbb", "ames", "del", "herg"]
    jobs = 20
    result_folder_path = Path("...")

    all_results = []

    for dataset in datasets:
        for job in range(1, jobs + 1):
            for rep in representations:
                auroc = load_auroc_values(dataset, job, rep)
                if auroc is not None:
                    all_results.append(
                        {"Dataset": dataset, "Representation": rep, "AUROC": auroc}
                    )

    # Convert results to a DataFrame
    auroc_df = pd.DataFrame(all_results)

    # Rename columns
    column_rename_map = {
        "transformer_matrix": "SMILES",
        "rdkit": "RDKitFP",
        "graph": "Graph",
    }
    auroc_df["Representation"] = auroc_df["Representation"].replace(column_rename_map)

    # Rename datasets
    dataset_rename_map = {"ames": "Ames", "bbb": "BBB", "del": "DEL", "herg": "hERG"}

    auroc_df["Dataset"] = auroc_df["Dataset"].replace(dataset_rename_map)

    # Create Figure
    y_min = np.floor(auroc_df["AUROC"].min() * 10) / 10  # Round down to nearest 0.1
    y_max = np.ceil(auroc_df["AUROC"].max() * 10) / 10  # Round up to nearest 0.1
    y_min -= 0.05
    y_max += 0.001
    # Set the style to white background
    sns.set_theme(style="whitegrid")
    # Create a figure with 4 subfigures
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Flatten the axes array for easier iteration
    axes = axes.flatten()

    # Get unique datasets
    datasets = auroc_df["Dataset"].unique()

    # Create a violinplot for each dataset
    for i, dataset in enumerate(datasets):
        data = auroc_df[auroc_df["Dataset"] == dataset]

        sns.violinplot(
            x="Representation",
            y="AUROC",
            hue="Representation",
            data=data,
            palette="Pastel1",
            legend=False,
            ax=axes[i],
        )

        axes[i].set_title(f"{dataset}", fontsize=14)
        axes[i].set_xlabel("")  # Remove x-axis label
        axes[i].set_ylabel("AUROC", fontsize=12)
        axes[i].tick_params(axis="both", which="major", labelsize=10)

        # Set the same y-axis limits for all subplots
        axes[i].set_ylim(y_min, y_max)

        # Remove top and right spines
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)

    # Adjust layout and save the figure
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.savefig(
        result_folder_path / "model_performance" / "datasets_comparison.pdf",
        bbox_inches="tight",
    )
    plt.close()
