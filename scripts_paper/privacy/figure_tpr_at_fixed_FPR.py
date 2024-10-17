import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from scipy.stats import wilcoxon
import numpy as np


def extract_highest_tpr(file_path: Path, target_fpr: float) -> float:
    df = pd.read_csv(file_path)
    if target_fpr == 0:
        fpr_df = df[df["fpr"] == 0]
    else:
        closest_fpr_idx = (df["fpr"] - target_fpr).abs().idxmin()
        closest_fpr = df.loc[closest_fpr_idx, "fpr"]
        fpr_df = df[df["fpr"] == closest_fpr]

    if not fpr_df.empty:
        return fpr_df["tpr"].max()
    else:
        return None


def significance_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return " "


def add_significance_stars(ax, data, baseline, x_col, y_col, hue_col):
    hue_order = data[hue_col].unique()
    x_order = data[x_col].unique()

    positions = []
    for attack in x_order:
        for hue in hue_order:
            subset = data[(data[x_col] == attack) & (data[hue_col] == hue)][y_col]
            if len(subset) > 0:
                stat, p = wilcoxon(subset - baseline, alternative="greater")
            else:
                p = 1  # Default to non-significant if no data

            stars = significance_stars(p)
            positions.append((attack, hue, stars))

    num_hues = len(hue_order)
    width = 0.8 / num_hues
    offsets = np.linspace(
        -(num_hues - 1) * width / 2, (num_hues - 1) * width / 2, num_hues
    )

    for attack_idx, attack in enumerate(x_order):
        for hue_idx, hue in enumerate(hue_order):
            x = attack_idx + offsets[hue_idx]
            stars = positions[attack_idx * len(hue_order) + hue_idx][2]
            ax.text(
                x,
                -0.01,
                stars,
                ha="center",
                va="top",
                fontsize=8,
                color="red",
                transform=ax.get_xaxis_transform(),
            )


def main():
    # adapt paths according to your folder layout
    base_dir = Path("...")
    representations = [
        "ECFP4",
        "ECFP6",
        "graph",
        "MACCS",
        "rdkit",
        "transformer_matrix",
    ]
    attacks = ["lira", "rmia"]
    datasets = ["bbb", "ames", "del", "herg"]
    target_fpr = 0  # 1e-3

    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # 2x2 grid of subplots
    axes = axes.flatten()

    for idx, dataset in enumerate(datasets):
        combined_data = []
        if dataset == "del":
            temp = "user_provided_dataset"
        else:
            temp = dataset
        dataset_size = len(
            pd.read_csv(base_dir / "..." / "data_dir" / "train.csv")
        )  # path to one of the training datasets (they have the same size between each repetition)
        baseline_tpr = 2 / dataset_size
        if target_fpr != 0:
            baseline_tpr = target_fpr
        for attack in attacks:
            for job_num in range(1, 21):
                job_folder = base_dir / dataset / f"job_{job_num}"

                for rep in representations:
                    if dataset == "del":
                        csv_file = (
                            job_folder
                            / "..."
                            / "privacy"
                            / "results"
                            / attack
                            / "ROC.csv"
                        )  # path to each MIA roc file

                    try:
                        tpr = extract_highest_tpr(csv_file, target_fpr)
                        if tpr is not None:
                            combined_data.append(
                                {"attack": attack, "representation": rep, "tpr": tpr}
                            )
                    except FileNotFoundError:
                        print(f"File not found: {csv_file}")
                    except Exception as e:
                        print(f"Error processing file {csv_file}: {e}")

        combined_df = pd.DataFrame(combined_data)

        combined_df["representation"] = combined_df["representation"].replace(
            {"transformer_matrix": "SMILES", "rdkit": "RDKitFP", "graph": "Graph"}
        )

        dataset_titles = {"ames": "Ames", "bbb": "BBB", "del": "DEL", "herg": "hERG"}

        ax = axes[idx]
        sns.boxplot(
            x="attack",
            y="tpr",
            hue="representation",
            data=combined_df,
            ax=ax,
            palette="Pastel1",
        )

        ax.axhline(y=baseline_tpr, color="grey", linestyle="--", label=f"Baseline")

        add_significance_stars(
            ax, combined_df, baseline_tpr, "attack", "tpr", "representation"
        )

        ax.set_xlabel("Attack", fontsize=10)
        ax.set_ylabel("TPR", fontsize=10)
        ax.set_ylim(bottom=0)
        ax.set_title(dataset_titles.get(dataset), fontsize=12, pad=10)
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.grid(True, axis="y")

        ax.set_xticklabels(
            [
                "LiRA" if label.get_text() == "lira" else "RMIA"
                for label in ax.get_xticklabels()
            ]
        )

        # Remove individual legends
        ax.legend().set_visible(False)

        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Collect handles and labels from the last axis
    handles, labels = ax.get_legend_handles_labels()

    # Add a common legend at the top
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=7,
        fontsize=10,
        title="Representation",
        title_fontsize=12,
        bbox_to_anchor=(0.5, 1.05),
    )

    plt.tight_layout()

    result_folder_path = Path(
        "..."
    )  # path to result folder where you want to save the figure
    result_folder_path.mkdir(exist_ok=True, parents=True)
    plt.savefig(
        result_folder_path / f"tpr_distribution_at_fpr_{target_fpr}.pdf",
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    main()
