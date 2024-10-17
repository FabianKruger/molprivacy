import pandas as pd
from pathlib import Path
from typing import List, Tuple
from scipy.stats import mannwhitneyu
from itertools import product
import numpy as np


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def calculate_label_distribution(
    population: pd.DataFrame,
    subset: pd.DataFrame,
    label_col: str,
    subset_label_col: str,
) -> Tuple[float, float]:
    subset_label_ratio = subset[subset_label_col].mean()
    population_label_ratio = population[label_col].mean()
    return subset_label_ratio, population_label_ratio


def create_dataframe(
    representations: List[str], dataset: str, base_path: Path
) -> pd.DataFrame:
    data = {
        "Dataset": [],
        "Representation": [],
        "Population_Label_Mean": [],
        "Population_Label_Std": [],
        "LIRA_Label_Mean": [],
        "LIRA_Label_Std": [],
        "LIRA_P_Value": [],
        "LIRA_Sample_Size": [],
        "RMIA_Label_Mean": [],
        "RMIA_Label_Std": [],
        "RMIA_P_Value": [],
        "RMIA_Sample_Size": [],
    }

    for rep in representations:
        population_data = []
        lira_data = []
        rmia_data = []
        for job in range(1, 21):
            try:
                pop_df = load_data(
                    base_path / dataset / f"job_{job}" / "data_dir" / "train.csv"
                )
                lira_df = load_data(
                    base_path
                    / dataset
                    / f"job_{job}"
                    / "..."
                    / "privacy"
                    / "results"
                    / "true_positives_at_FPR0"
                    / "lira.csv"
                )
                rmia_df = load_data(
                    base_path
                    / dataset
                    / f"job_{job}"
                    / "..."
                    / "privacy"
                    / "results"
                    / "true_positives_at_FPR0"
                    / "rmia.csv"
                )
                population_data.append(pop_df)
                lira_data.append(lira_df)
                rmia_data.append(rmia_df)
            except Exception as e:
                print(f"Exception for {dataset} {job} {rep}: {e}")

        population_labels = [df["label"].mean() for df in population_data]
        lira_labels = [
            calculate_label_distribution(pop_df, leak_df, "label", "property_label")[0]
            for pop_df, leak_df in zip(population_data, lira_data)
        ]
        rmia_labels = [
            calculate_label_distribution(pop_df, leak_df, "label", "property_label")[0]
            for pop_df, leak_df in zip(population_data, rmia_data)
        ]

        def safe_mannwhitneyu(x, y):
            x = [val for val in x if not np.isnan(val) and not np.isinf(val)]
            y = [val for val in y if not np.isnan(val) and not np.isinf(val)]
            if (
                len(x) > 1
                and len(y) > 1
                and not (all(v == x[0] for v in x) and all(v == y[0] for v in y))
            ):
                return mannwhitneyu(x, y).pvalue
            return np.nan

        data["Dataset"].append(dataset)
        data["Representation"].append(rep)
        data["Population_Label_Mean"].append(pd.Series(population_labels).mean())
        data["Population_Label_Std"].append(pd.Series(population_labels).std())
        data["LIRA_Label_Mean"].append(pd.Series(lira_labels).mean())
        data["LIRA_Label_Std"].append(pd.Series(lira_labels).std())
        data["LIRA_P_Value"].append(safe_mannwhitneyu(lira_labels, population_labels))
        data["LIRA_Sample_Size"].append(len(lira_labels))
        data["RMIA_Label_Mean"].append(pd.Series(rmia_labels).mean())
        data["RMIA_Label_Std"].append(pd.Series(rmia_labels).std())
        data["RMIA_P_Value"].append(safe_mannwhitneyu(rmia_labels, population_labels))
        data["RMIA_Sample_Size"].append(len(rmia_labels))

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    base_path = Path("...")
    datasets = ["ames", "bbb", "del", "herg"]
    representations = [
        "ECFP4",
        "ECFP6",
        "graph",
        "MACCS",
        "rdkit",
        "transformer_matrix",
    ]
    output_folder = Path("...")
    output_folder.mkdir(parents=True, exist_ok=True)

    all_data = []
    for dataset in datasets:
        df = create_dataframe(representations, dataset, base_path)
        all_data.append(df)

    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv(output_folder / "label_distribution_summary.csv", index=False)
