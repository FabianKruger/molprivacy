import pandas as pd
import numpy as np
from pathlib import Path

def extract_highest_tpr(file_path: Path, target_fpr: float) -> float:
    """Extract the highest TPR value for the closest FPR to the target FPR in the given file."""
    df = pd.read_csv(file_path)
    fpr_df = df[df['fpr'] == target_fpr]
    
    if not fpr_df.empty:
        return fpr_df['tpr'].max()
    else:
        return None

def main():
    base_dir = Path("...")
    representations = ["ECFP4", "ECFP6", "graph", "MACCS", "rdkit", "transformer_matrix"]
    attacks = ["lira", "rmia"]
    datasets = ["bbb", "ames", "del", "herg"]

    fpr_threshold = 0  # The specific FPR threshold to compare

    # Initialize counters for graph representation analysis
    graph_lowest_count = 0
    dataset_percentage_lowers = []

    for dataset in datasets:
        dataset_graph_median_values = []
        dataset_other_median_values = []

        # Store TPR values for each representation and attack
        tpr_values = {attack: {rep: [] for rep in representations} for attack in attacks}

        for rep in representations:
            for attack in attacks:
                tpr_list = []

                for job_num in range(1, 21):
                    job_folder = base_dir / dataset / f"job_{job_num}"

                    csv_file = job_folder / f"..." / "privacy" / "results" / attack / "ROC.csv"

                    try:
                        tpr_value = extract_highest_tpr(csv_file, fpr_threshold)
                        if tpr_value is not None:
                            tpr_list.append(tpr_value)
                    except FileNotFoundError:
                        print(f"File not found: {csv_file}")
                    except Exception as e:
                        print(f"Error processing file {csv_file}: {e}")

                # Store the TPR values for further analysis
                if tpr_list:
                    tpr_values[attack][rep] = tpr_list

        # Analyze TPR values per dataset
        for attack in attacks:
            graph_tpr = np.median(tpr_values[attack]["graph"])
            other_medians = [np.median(tpr_values[attack][rep]) for rep in representations if rep != "graph"]
            avg_other_median = np.mean(other_medians)  # Average of the medians for other representations

            # Check if "graph" has the lowest TPR
            if graph_tpr == min([graph_tpr] + other_medians):
                graph_lowest_count += 1

            # Store median values for calculating percentage lower
            dataset_graph_median_values.append(graph_tpr)
            dataset_other_median_values.append(avg_other_median)

        # Calculate the percentage lower for this dataset
        if dataset_other_median_values:
            dataset_avg_graph_median = np.mean(dataset_graph_median_values)
            dataset_avg_other_median = np.mean(dataset_other_median_values)
            
            if dataset_avg_other_median > 0:
                dataset_percentage_lower = (dataset_avg_other_median - dataset_avg_graph_median) / dataset_avg_other_median * 100
                dataset_percentage_lowers.append(dataset_percentage_lower)

    # Calculate overall statistics
    if dataset_percentage_lowers:
        mean_percentage_lower = np.mean(dataset_percentage_lowers)
        std_percentage_lower = np.std(dataset_percentage_lowers)

        print(f"'Graph' representation had the lowest TPR in {graph_lowest_count} cases out of {len(datasets) * len(attacks)} total cases.")
        print(f"The median TPR for 'graph' was, on average, {mean_percentage_lower:.2f}% lower than the average of medians TPR of all other representations across all datasets (STD: {std_percentage_lower:.2f}%).")
    else:
        print("No data available to calculate statistics for 'graph' representation.")

if __name__ == "__main__":
    main()
