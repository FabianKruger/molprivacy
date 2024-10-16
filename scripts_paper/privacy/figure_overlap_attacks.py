import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import numpy as np



# Load CSV files
def load_csv_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def calculate_overlap(subset1: pd.DataFrame, subset2: pd.DataFrame, population: pd.DataFrame):
    common_molecules = pd.merge(subset1, subset2, on='molecules')
    num_common_molecules = common_molecules.shape[0]
    num_non_common_molecules = subset1.shape[0] + subset2.shape[0] - 2 * common_molecules.shape[0]

    if subset1.shape[0] != 0 and subset2.shape[0] != 0:
        mu = (subset1.shape[0] * subset2.shape[0]) / population.shape[0]
        percentage_of_possible_overlap = common_molecules.shape[0] / min(subset1.shape[0], subset2.shape[0])
    else:
        percentage_of_possible_overlap = None
        mu = None

    return {
        "number_common_molecules": num_common_molecules,
        "number_non_common_molecules": num_non_common_molecules,
        "mu": mu,
        "percentage_of_possible_overlap": percentage_of_possible_overlap,
    }

def significance_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''

def add_significance_stars(ax, data, x_col, y_col, hue_col, test_col):
    hue_order = data[hue_col].unique()
    x_order = data[x_col].unique()
    
    y_ticks = ax.get_yticks()
    y_tick_distance = y_ticks[1] - y_ticks[0]

    num_hues = len(hue_order)
    width = 0.8 / num_hues
    offsets = np.linspace(-(num_hues - 1) * width / 2, (num_hues - 1) * width / 2, num_hues)

    for dataset_index, dataset in enumerate(x_order):
        for hue_index, rep in enumerate(hue_order):
            subset = data[(data[x_col] == dataset) & (data[hue_col] == rep)]
            if len(subset) > 0:
                _, p = wilcoxon(subset[test_col], alternative="greater")
                stars = significance_stars(p)
                if stars:
                    x = dataset_index + offsets[hue_index]
                    y = -y_tick_distance/2.5
                    ax.text(x, y, stars, ha='center', va='bottom', fontsize=12, color='red')

if __name__ == "__main__":
    # adapt paths to your folder structure
    folder_path = Path("...")
    plot_folder_path = Path("...")
    datasets = ["bbb", "ames", "del", "herg"]
    representations = ["ECFP4", "ECFP6", "graph", "MACCS", "rdkit", "transformer_matrix"]

    all_results = []

    for dataset in datasets:
        for representation in representations:
            for job in range(1, 21):
                job_path = folder_path / dataset / f"job_{job}"
                try:
                    subfolder_path = job_path / "..." / "privacy" / "results" / "true_positives_at_FPR0"
                    lira_path = subfolder_path / "lira.csv"
                    rmia_path = subfolder_path / "rmia.csv"
                    population_path = job_path / "data_dir" / "train.csv"

                    lira = load_csv_file(lira_path)
                    rmia = load_csv_file(rmia_path)
                    population = load_csv_file(population_path)

                    result = calculate_overlap(lira, rmia, population)
                    if result["number_common_molecules"] is not None and result["mu"] is not None:
                        deviation = result["number_common_molecules"] - result["mu"]
                        all_results.append({
                            "dataset": dataset,
                            "representation": representation,
                            "deviation": deviation,
                            "overlap": result["percentage_of_possible_overlap"]
                        })
                except Exception as e:
                    print(f"Error processing job {job} for dataset {dataset} and representation {representation}: {e}")

    # Create DataFrame from all results
    plot_df = pd.DataFrame(all_results)

    plot_df['representation'] = plot_df['representation'].replace({
        "transformer_matrix": "SMILES",
        "rdkit": "RDKitFP",
        "graph": "Graph"
    })
    
    # Set the style to match the TPR plot
    sns.set_theme(style="whitegrid")

    # Create a single figure
    plt.figure(figsize=(10, 6))

    # Create the boxplot
    ax = sns.boxplot(data=plot_df, x='dataset', y='overlap', hue='representation', 
                    palette="Pastel1", width=0.8, fliersize=3)

    # Customize the plot
    plt.xlabel('Dataset', fontsize=14)
    plt.ylabel('Proportion of Possible Overlap', fontsize=14)

    # Set y-axis limits and ticks
    plt.ylim(-0.005, 1.005)  # Adjusted to show full whiskers
    plt.yticks(np.arange(0, 1.1, 0.1))

    # Customize x-axis labels
    dataset_labels = {'ames': 'Ames', 'bbb': 'BBB', 'del': 'DEL', 'herg': 'hERG'}
    ax.set_xticklabels([dataset_labels.get(label.get_text(), label.get_text()) for label in ax.get_xticklabels()])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # Add significance stars
    add_significance_stars(ax, plot_df, 'dataset', 'overlap', 'representation', 'deviation')

    # Customize legend - place inside the figure
    plt.legend(title='Representation', loc='upper right', fontsize=10)

    # Adjust layout and save
    plt.tight_layout()
    plot_file_path = plot_folder_path / "percentage_possible_overlap_attacks.pdf"
    plt.savefig(plot_file_path, bbox_inches='tight')
    plt.close()


    # Plotting the deviation data
    plt.figure(figsize=(10, 6))

    # Create the boxplot
    ax = sns.boxplot(data=plot_df, x='dataset', y='deviation', hue='representation',
                    palette="Pastel1", width=0.8, fliersize=3)

    # Customize the plot
    plt.xlabel('Dataset', fontsize=14)
    plt.ylabel('Deviation from Expected Overlap', fontsize=14)

    # Customize x-axis labels
    dataset_labels = {'ames': 'Ames', 'bbb': 'BBB', 'del': 'DEL', 'herg': 'hERG'}
    ax.set_xticklabels([dataset_labels.get(label.get_text(), label.get_text()) for label in ax.get_xticklabels()])

    # Remove top, right, and bottom spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Add significance stars based on the deviation
    add_significance_stars(ax, plot_df, 'dataset', 'deviation', 'representation', 'deviation')

    # Customize legend - place inside the figure
    plt.legend(title='Representation', loc='upper right', fontsize=10)

    # Adjust y-axis limits to show full whiskers
    y_min, y_max = plt.ylim()
    plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))

    # Adjust layout and save
    plt.tight_layout()
    plot_file_path = plot_folder_path / "deviation_overlaps_attacks_from_random.pdf"
    plt.savefig(plot_file_path, bbox_inches='tight')
    plt.close()