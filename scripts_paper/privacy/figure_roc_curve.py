import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def main():
    base_dir = Path("...")
    representations = ["ECFP4", "ECFP6", "graph", "MACCS", "rdkit", "transformer_matrix"]
    attacks = ["lira", "rmia"]
    datasets = ["bbb", "ames", "del", "herg"]

    # Rename mappings
    rep_rename = {
        "transformer_matrix": "SMILES",
        "rdkit": "RDKitFP",
        "graph": "Graph"
    }

    dataset_titles = {
        "ames": "Ames",
        "bbb": "BBB",
        "del": "DEL",
        "herg": "hERG"
    }

    attack_titles = {
        "lira": "LiRA",
        "rmia": "RMIA"
    }

    # Set the color palette to Pastel1
    sns.set_palette("Pastel1")
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(len(datasets), len(attacks), figsize=(16, 20))
    axes = axes.reshape(len(datasets), len(attacks))

    # Define the FPR ranges for each dataset
    fpr_ranges = {
        "bbb": np.logspace(-3, 0, 100),  # 10^-3 to 1
        "ames": np.logspace(-4, 0, 100),  # 10^-4 to 1
        "del": np.logspace(-5, 0, 100),  # 10^-5 to 1
        "herg": np.logspace(-6, 0, 100),  # 10^-6 to 1
    }

    for idx, dataset in enumerate(datasets):

        common_fpr = fpr_ranges[dataset]  # Select the appropriate FPR range

        for jdx, attack in enumerate(attacks):
            ax = axes[idx, jdx]
            has_valid_data = False  # Flag to check if there's any valid data

            for rep in representations:
                tpr_list = []

                for job_num in range(1, 21):
                    job_folder = base_dir / dataset / f"job_{job_num}"
                    
                    csv_file = job_folder / "..." / "privacy" / "results" / attack / "ROC.csv" # path to roc csv file

                    try:
                        df = pd.read_csv(csv_file)
                        fpr = df['fpr'].values
                        tpr = df['tpr'].values

                        # Ensure that FPR and TPR are sorted in ascending order
                        if fpr[0] > fpr[-1]:  # Check if sorted in descending order
                            fpr = fpr[::-1]  # Reverse the order
                            tpr = tpr[::-1]  # Reverse the order

                        if np.any(tpr > 0):
                            has_valid_data = True

                        # Interpolate TPR values at the common FPR points
                        interp_tpr = np.interp(common_fpr, fpr, tpr)
                        tpr_list.append(interp_tpr)
                    except FileNotFoundError:
                        print(f"File not found: {csv_file}")
                    except Exception as e:
                        print(f"Error processing file {csv_file}: {e}")

                # Convert lists to arrays for easier manipulation
                if tpr_list:
                    tpr_array = np.array(tpr_list)

                    # Calculate the mean
                    mean_tpr = np.median(tpr_array, axis=0)

                    # Plot the mean ROC curve using Seaborn
                    sns.lineplot(x=common_fpr, y=mean_tpr, ax=ax, label=rep_rename.get(rep, rep))

            if has_valid_data:
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlim(common_fpr[0], common_fpr[-1])  # Set x-axis limits
            else:
                ax.set_xscale('log')
                ax.set_yscale('linear')  # Default to linear scale if no valid data

            # Plot the diagonal dashed line without adding it to the legend
            ax.plot([common_fpr[0], 1], [common_fpr[0], 1], linestyle='--', color='gray', label='_nolegend_')

            # Set the title using the renamed dataset and attack names
            ax.set_title(f"{dataset_titles.get(dataset, dataset)} - {attack_titles.get(attack, attack)}")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc='lower right')

    plt.tight_layout()
    result_folder_path = Path("...") # path to folder where you want to save the figure
    result_folder_path.mkdir(exist_ok=True)
    plt.savefig(result_folder_path / f"roc_curves.pdf", bbox_inches='tight')
    plt.close()
    
if __name__ == "__main__":
    main()
