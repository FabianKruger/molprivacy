import pandas as pd
from pathlib import Path
from rdkit import Chem
from typing import List, Tuple, Any
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from itertools import product
from tqdm import tqdm
import pickle


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def get_atom_counts(smiles: str) -> int:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid SMILES string")
    return molecule.GetNumAtoms()


def subset_mol_sizes_from_population(
    population: List[str], subset: List[str]
) -> Tuple[List[int], List[int]]:
    subset_sizes = []
    complement_sizes = []
    for ele in population:
        size = get_atom_counts(ele)
        if ele in subset:
            subset_sizes.append(size)
        else:
            complement_sizes.append(size)
    return subset_sizes, complement_sizes


def create_dataframe(
    representations: List[str],
    dataset: str,
    attack: str,
    base_path: Path,
    temp_folder: Path,
) -> pd.DataFrame:
    data = {"Representation": [], "Set": [], "Size": []}

    # Check if saved data exists
    saved_data_path = temp_folder / f"{dataset}_{attack}_sizes.pkl"
    if saved_data_path.exists():
        with open(saved_data_path, "rb") as f:
            return pickle.load(f)

    for rep in representations:
        population_data = []
        leakage_data = []
        for job in range(1, 21):
            try:
                pop_df = load_data(
                    base_path / dataset / f"job_{job}" / "data_dir" / "train.csv"
                )
                leak_df = load_data(
                    base_path
                    / dataset
                    / f"job_{job}"
                    / "..."
                    / "privacy"
                    / "results"
                    / "true_positives_at_FPR0"
                    / f"{attack}.csv"
                )

                population_data.append(pop_df["smiles"].tolist())
                leakage_data.append(leak_df["molecules"].tolist())
            except Exception as e:
                print(f"Exception for {dataset} {job} {rep}: {e}")

        for pop, leaked in zip(population_data, leakage_data):
            sub_sizes, pop_sizes = subset_mol_sizes_from_population(pop, leaked)
            data["Representation"].extend([rep] * (len(sub_sizes) + len(pop_sizes)))
            data["Set"].extend(
                ["Subset"] * len(sub_sizes) + ["Population"] * len(pop_sizes)
            )
            data["Size"].extend(sub_sizes + pop_sizes)

    df = pd.DataFrame(data)

    # Save the calculated sizes
    temp_folder.mkdir(parents=True, exist_ok=True)
    with open(saved_data_path, "wb") as f:
        pickle.dump(df, f)

    return df


def significance_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return " "


def plot_dataframe(
    dfs: List[pd.DataFrame],
    figure_folder: Path,
    attacks: List[str],
    datasets: List[str],
) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(20, 30))

    representation_mapping = {
        "transformer_matrix": "SMILES",
        "rdkit": "RDKitFP",
        "graph": "Graph",
    }

    dataset_titles = {"ames": "Ames", "bbb": "BBB", "del": "DEL", "herg": "hERG"}

    # Updated set labels
    set_labels = {"Subset": "Identified molecules", "Population": "Training data"}

    for row, dataset in enumerate(datasets):
        for col, attack in enumerate(attacks):
            ax = axes[row, col]
            df = dfs[row * len(attacks) + col]

            df["Representation"] = df["Representation"].replace(representation_mapping)
            df["Set"] = df["Set"].replace(
                set_labels
            )  # Update the labels in the dataframe

            sns_plot = sns.boxplot(
                data=df,
                x="Representation",
                y="Size",
                hue="Set",
                ax=ax,
                showfliers=False,
            )

            unique_reps = df["Representation"].unique()
            for i, rep in enumerate(unique_reps):
                subset_sizes = df[
                    (df["Representation"] == rep)
                    & (df["Set"] == "Identified molecules")
                ]["Size"]
                population_sizes = df[
                    (df["Representation"] == rep) & (df["Set"] == "Training data")
                ]["Size"]
                stat, p_value = mannwhitneyu(subset_sizes, population_sizes)

                stars = significance_stars(p_value)
                if stars != " ":
                    y_min = df[df["Representation"] == rep]["Size"].min()
                    y_range = df["Size"].max() - df["Size"].min()
                    star_y = y_min - 0.05 * y_range

                    ax.text(
                        i,
                        star_y,
                        stars,
                        ha="center",
                        va="top",
                        color="red",
                        fontsize=16,
                    )

            ax.set_ylim(-5, 65)  # Set y-axis limits from -5 to 65

            ax.set_title(f"{dataset_titles[dataset]} - {attack.upper()}")
            ax.set_xlabel("Representation")
            ax.set_ylabel("Atom count")
            ax.legend(title="Set", loc="upper right")
            ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    save_path = figure_folder / "molecule_sizes_boxplot.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    base_path = Path("...")
    datasets = ["bbb", "ames", "del", "herg"]
    attacks = ["lira", "rmia"]
    representations = [
        "ECFP4",
        "ECFP6",
        "graph",
        "MACCS",
        "rdkit",
        "transformer_matrix",
    ]
    figure_folder = Path("...")
    temp_folder = Path(
        "..."
    )  # to save sizes in case of crashing (script takes some time to run)

    all_dfs = []
    for dataset, attack in tqdm(product(datasets, attacks)):
        df = create_dataframe(representations, dataset, attack, base_path, temp_folder)
        all_dfs.append(df)

    plot_dataframe(all_dfs, figure_folder, attacks, datasets)
