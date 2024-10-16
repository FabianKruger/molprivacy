import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List
from itertools import combinations

# Load CSV files
def load_csv_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def calculate_overlap_percentage(subset1: pd.DataFrame, subset2: pd.DataFrame):
    common_molecules = pd.merge(subset1, subset2, on='molecules')
    if subset1.shape[0] != 0 and subset2.shape[0] != 0:
        percentage_of_possible_overlap = common_molecules.shape[0] / min(subset1.shape[0], subset2.shape[0])
    else:
        percentage_of_possible_overlap = None
    return percentage_of_possible_overlap

def plot_combined_boxplots(datasets: List[str], representations: List[str], attacks: List[str]):
    general_path = Path("...")
    fig, axes = plt.subplots(4, 2, figsize=(20, 40))

    representation_names = {
        "transformer_matrix": "SMILES",
        "rdkit": "RDKitFP",
        "graph": "Graph",
        "MACCS": "MACCS"
    }
    dataset_titles = {
        "ames": "Ames",
        "bbb": "BBB",
        "del": "DEL",
        "herg": "hERG"
    }

    for idx, (dataset, attack) in enumerate([(d, a) for d in datasets for a in attacks]):
        results = {pair: [] for pair in combinations(representations, 2)}
        
        for job in range(1, 21):
            for rep1, rep2 in combinations(representations, 2):
                subset_1_path = general_path / dataset / f"job_{job}" / "..." / "privacy" / "results" / "true_positives_at_FPR0" / f"{attack}.csv"
                subset_2_path = general_path / dataset / f"job_{job}" / "..." / "privacy" / "results" / "true_positives_at_FPR0" / f"{attack}.csv"
                try:    
                    subset_1 = load_csv_file(subset_1_path)
                    subset_2 = load_csv_file(subset_2_path)
                    percentage_overlap = calculate_overlap_percentage(subset_1, subset_2)
                    results[(rep1, rep2)].append(percentage_overlap)
                except FileNotFoundError as e:
                    print(e)
        
        data = []
        for (rep1, rep2), overlap_values in results.items():
            overlap_values = [v for v in overlap_values if v is not None]
            for value in overlap_values:
                data.append([f"{representation_names.get(rep1, rep1)} vs {representation_names.get(rep2, rep2)}", value])
        
        df = pd.DataFrame(data, columns=['Representation Pair', 'Overlap'])
        
        ax = axes[idx // 2, idx % 2]
        sns.boxplot(x='Representation Pair', y='Overlap', data=df, ax=ax, color='lightgrey', medianprops=dict(color="black"))
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=14)
        ax.set_title(f'{dataset_titles[dataset]}; {attack.upper()}', fontsize=18)
        ax.set_ylabel('Percentage of Possible Overlap', fontsize=16)
        ax.set_xlabel('', fontsize=16)
        ax.set_ylim(0, 1.005)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Increase tick label size
        ax.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    output_dir = Path("...")
    plt.savefig(output_dir / "overlap_representations.pdf", bbox_inches='tight', dpi=300)
    plt.close()

def main(datasets: List[str], representations: List[str], attacks: List[str]):
    plot_combined_boxplots(datasets, representations, attacks)

if __name__=="__main__":
    datasets = ["bbb", "ames", "del", "herg"]
    representations = ["ECFP4", "ECFP6", "graph", "transformer_matrix", "rdkit", "MACCS"]
    attacks = ["lira", "rmia"]
    main(datasets, representations, attacks)