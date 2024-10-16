import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon

def extract_highest_tpr(file_path: Path, target_fpr: float) -> float:
    """Extract the highest TPR value for the closest FPR to the target FPR in the given file."""
    df = pd.read_csv(file_path)
    if target_fpr == 0:
        fpr_df = df[df['fpr'] == 0]
    else:
        closest_fpr_idx = (df['fpr'] - target_fpr).abs().idxmin()
        closest_fpr = df.loc[closest_fpr_idx, 'fpr']
        fpr_df = df[df['fpr'] == closest_fpr]
    
    if not fpr_df.empty:
        return fpr_df['tpr'].max()
    else:
        return None

def main():
    base_dir = Path("...")
    representations = ["ECFP4", "ECFP6", "graph", "MACCS", "rdkit", "transformer_matrix"]
    attacks = ["lira", "rmia"]
    datasets = ["bbb", "ames", "del", "herg"]

    fpr_threshold = 10**-3  # The specific FPR threshold to compare

    # Number of tests for Bonferroni correction
    num_tests = len(datasets) * len(representations)
    alpha = 0.05  # Original significance level
    corrected_alpha = alpha / num_tests  # Bonferroni corrected significance level

    # Initialize lists to collect data for the final table
    results = []

    # Initialize summary counts
    significance_summary = {dataset: {"LiRA is statistically significantly higher": 0, 
                                      "RMIA is statistically significantly higher": 0, 
                                      "No significant difference was found": 0} for dataset in datasets}

    for dataset in datasets:
        # Store TPR values for each representation and attack
        tpr_values = {attack: {rep: [] for rep in representations} for attack in attacks}

        for rep in representations:
            for attack in attacks:
                tpr_list = []

                for job_num in range(1, 21):
                    job_folder = base_dir / dataset / f"job_{job_num}"

                    csv_file = job_folder / "..." / "privacy" / "results" / attack / "ROC.csv"

                    try:
                        tpr_value = extract_highest_tpr(csv_file, fpr_threshold)
                        if tpr_value is not None:
                            tpr_list.append(tpr_value)
                    except FileNotFoundError:
                        print(f"File not found: {csv_file}")
                    except Exception as e:
                        print(f"Error processing file {csv_file}: {e}")

                # Store the TPR values for statistical testing
                if tpr_list:
                    tpr_values[attack][rep] = tpr_list

        # Perform Wilcoxon signed-rank test and compare median TPR values between 'rmia' and 'lira'
        for rep in representations:
            lira_stat_sig = 0
            rmia_stat_sig = 0
            no_significant_diff = 0

            if tpr_values['rmia'][rep] and tpr_values['lira'][rep]:
                # Perform the Wilcoxon signed-rank test
                try:
                    stat, p_value = wilcoxon(tpr_values['rmia'][rep], tpr_values['lira'][rep])
                    # Apply Bonferroni correction for multiple testing
                    if p_value < corrected_alpha:
                        if np.median(tpr_values['rmia'][rep]) > np.median(tpr_values['lira'][rep]):
                            rmia_stat_sig += 1
                            significance_summary[dataset]["RMIA is statistically significantly higher"] += 1
                        else:
                            lira_stat_sig += 1
                            significance_summary[dataset]["LiRA is statistically significantly higher"] += 1
                    else:
                        no_significant_diff += 1
                        significance_summary[dataset]["No significant difference was found"] += 1
                except ValueError:
                    # Handle the case where there are not enough paired differences
                    no_significant_diff += 1
                    significance_summary[dataset]["No significant difference was found"] += 1
            else:
                no_significant_diff += 1
                significance_summary[dataset]["No significant difference was found"] += 1

            # Append the results to the list
            results.append({
                "Dataset": dataset,
                "Representation": rep,
                "LiRA is statistically significantly higher": lira_stat_sig,
                "RMIA is statistically significantly higher": rmia_stat_sig,
                "No significant difference was found": no_significant_diff
            })

    # Convert the results to a DataFrame and print it
    # results_df = pd.DataFrame(results)
    # print("Detailed Results with Statistical Significance (Bonferroni Correction Applied):")
    # print(results_df)

    # Print summary of statistical significance for each dataset
    print("\nSummary of Statistical Significance for Each Dataset (Bonferroni Correction Applied):")
    for dataset, counts in significance_summary.items():
        print(f"Dataset '{dataset}':")
        print(f"  LiRA is statistically significantly higher in {counts['LiRA is statistically significantly higher']} representations.")
        print(f"  RMIA is statistically significantly higher in {counts['RMIA is statistically significantly higher']} representations.")
        print(f"  No significant difference was found in {counts['No significant difference was found']} representations.")

if __name__ == "__main__":
    main()
