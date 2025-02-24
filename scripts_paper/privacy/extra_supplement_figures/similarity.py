import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from scipy.stats import mannwhitneyu
from typing import Literal

# -------------------------
# Hyperparameters
# -------------------------
# Choose dataset: "ames" or "bbb"
dataset: Literal["ames", "bbb"] = "bbb"  
# Choose similarity_mode: "average" or "nearest"
similarity_mode: Literal["average", "nearest"] = "average"
# Instead of a single leak_source, we now compare both:
leak_sources = ["lira", "rmia"]

# -------------------------
# Set base directory and training file name based on dataset
# -------------------------
if dataset == "ames":
    base_dir = '...'
    train_filename = "ames.csv"
elif dataset == "bbb":
    base_dir = '...'
    train_filename = "train.csv"
else:
    raise ValueError("dataset must be either 'ames' or 'bbb'")

# -------------------------
# Define mapping dictionaries for short names.
# These mappings are used for plot titles.
fp_mapping = {
    "ECFP_4": "ECFP4",
    "ECFP_6": "ECFP6",
    "transformer_matrix": "SMILES",
    "rdkit": "RDKitFP",
    "graph": "Graph",
    "MACCS": "MACCS"
}
assay_mapping = {
    "ames": "Ames",
    "bbb": "BBB",
    "del": "DEL",
    "herg": "hERG"
}
leak_mapping = {
    "lira": "LiRA",
    "rmia": "RMIA"
}

# -------------------------
# Create the list of results folder names based on dataset.
# -------------------------
results_types = [
    f"results_{dataset}_ECFP_4",
    f"results_{dataset}_ECFP_6",
    f"results_{dataset}_graph",
    f"results_{dataset}_MACCS",
    f"results_{dataset}_rdkit",
    f"results_{dataset}_transformer_matrix"
]

# -------------------------
# Get assay from the base_dir (should be "ames" or "bbb")
# -------------------------
assay = os.path.basename(base_dir).lower()

# -------------------------
# Prepare a list to store statistical test results.
# -------------------------
stats_results = []

# -------------------------
# Prepare a figure with 6 subplots (2 rows x 3 columns).
# -------------------------
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes = axes.flatten()

# -------------------------
# Loop over each results folder.
# -------------------------
for idx, results_type in enumerate(results_types):
    combined_data_all = []  # Will hold data for both leak sources for this results_type

    # Process each leak source.
    for leak in leak_sources:
        processed_filename = f"{dataset}_train_similiarity_both_{leak}_{results_type}.csv"
        leak_data = []  # To accumulate data from job folders for this leak source
        
        # Loop over job folders job_1 to job_20.
        for job_num in range(1, 21):
            job_folder = f"job_{job_num}"
            data_dir = os.path.join(base_dir, job_folder, "data_dir")
            proc_file = os.path.join(data_dir, processed_filename)
            valid_file = False

            if os.path.exists(proc_file):
                try:
                    df = pd.read_csv(proc_file)
                    required_cols = {"average tanimoto similarity", "nearest tanimoto similarity", "leaked"}
                    if required_cols.issubset(df.columns):
                        valid_file = True
                        print(f"Loaded existing file for {job_folder} ({results_type}, {leak}): {processed_filename}")
                    else:
                        print(f"{proc_file} is missing required columns.")
                except Exception as e:
                    print(f"Error reading {proc_file}: {e}")

            if not valid_file:
                original_file = os.path.join(data_dir, train_filename)
                if not os.path.exists(original_file):
                    print(f"Original file not found: {original_file}. Skipping {job_folder}.")
                    continue

                df = pd.read_csv(original_file)
                if "smiles" not in df.columns:
                    print(f"'smiles' column not found in {original_file}. Skipping {job_folder}.")
                    continue

                # Compute fingerprints once.
                fingerprints = []
                for smi in df["smiles"]:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        fingerprints.append(None)
                    else:
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                        fingerprints.append(fp)

                avg_sim = []
                nearest_sim = []
                n = len(fingerprints)
                for i, fp in enumerate(fingerprints):
                    if fp is None:
                        avg_sim.append(None)
                        nearest_sim.append(None)
                        continue
                    other_fps = [fingerprints[j] for j in range(n) if j != i and fingerprints[j] is not None]
                    if not other_fps:
                        avg_sim.append(None)
                        nearest_sim.append(None)
                        continue
                    sims = DataStructs.BulkTanimotoSimilarity(fp, other_fps)
                    avg_sim.append(sum(sims)/len(sims))
                    nearest_sim.append(max(sims))
                df["average tanimoto similarity"] = avg_sim
                df["nearest tanimoto similarity"] = nearest_sim

                # Choose the label file based on leak source and results_type.
                if leak == "lira":
                    label_file = os.path.join(
                        base_dir,
                        job_folder,
                        results_type,
                        "leakpro_output",
                        "results",
                        "true_positives",
                        "lira.csv"
                    )
                elif leak == "rmia":
                    label_file = os.path.join(
                        base_dir,
                        job_folder,
                        results_type,
                        "leakpro_output",
                        "results",
                        "true_positives",
                        "rmia.csv"
                    )
                else:
                    raise ValueError("Leak source must be either 'lira' or 'rmia'")

                if os.path.exists(label_file):
                    try:
                        df_label = pd.read_csv(label_file)
                        leaked_set = set(df_label["molecules"].astype(str))
                    except Exception as e:
                        print(f"Error reading {label_file}: {e}")
                        leaked_set = set()
                else:
                    print(f"{label_file} not found for {job_folder}. Labeling all as not leaked (0).")
                    leaked_set = set()

                df["leaked"] = df["smiles"].astype(str).apply(lambda x: 1 if x in leaked_set else 0)

                df.to_csv(proc_file, index=False)
                print(f"Generated processed file for {job_folder} at {proc_file}")

            # Append required columns and add a column for leak_source.
            if {"average tanimoto similarity", "nearest tanimoto similarity", "leaked"}.issubset(df.columns):
                df["leak_source"] = leak_mapping.get(leak, leak)
                leak_data.append(df[["average tanimoto similarity", "nearest tanimoto similarity", "leaked", "leak_source"]])
            else:
                print(f"Skipping {job_folder} because required columns are missing.")

        if leak_data:
            leak_df = pd.concat(leak_data, ignore_index=True)
        else:
            print(f"No valid data for {results_type} and leak source {leak}.")
            leak_df = pd.DataFrame(columns=["average tanimoto similarity", "nearest tanimoto similarity", "leaked", "leak_source"])
        
        # Perform statistical test for this leak source.
        sim_col = "average tanimoto similarity" if similarity_mode == "average" else "nearest tanimoto similarity"
        group0 = leak_df.loc[leak_df["leaked"] == 0, sim_col].dropna()
        group1 = leak_df.loc[leak_df["leaked"] == 1, sim_col].dropna()
        if len(group0) > 0 and len(group1) > 0:
            try:
                U_stat, p_value = mannwhitneyu(group1, group0, alternative="less")
            except Exception as e:
                print("Error in mannwhitneyu test:", e)
                U_stat, p_value = None, None
        else:
            U_stat, p_value = None, None
        median0 = group0.median() if not group0.empty else None
        median1 = group1.median() if not group1.empty else None
        n0 = len(group0)
        n1 = len(group1)
        stats_results.append({
            "results_type": results_type,
            "leak_source": leak_mapping.get(leak, leak),
            "assay": assay_mapping.get(assay, assay),
            "similarity_mode": similarity_mode,
            "group0_n": n0,
            "group1_n": n1,
            "median_non_leaked": median0,
            "median_leaked": median1,
            "U_statistic": U_stat,
            "p_value": p_value
        })
        
        combined_data_all.append(leak_df)
    
    # Combine data from both leak sources.
    if combined_data_all:
        combined_df = pd.concat(combined_data_all, ignore_index=True)
    else:
        print(f"No valid data available for plotting for {results_type}.")
        combined_df = pd.DataFrame(columns=["average tanimoto similarity", "nearest tanimoto similarity", "leaked", "leak_source"])

    # -------------------------
    # Binning and plotting logic.
    # -------------------------
    sim_col = "average tanimoto similarity" if similarity_mode == "average" else "nearest tanimoto similarity"
    if similarity_mode == "average":
        try:
            combined_df["similarity_bin"], bins = pd.qcut(combined_df[sim_col],
                                                        q=10, retbins=True, duplicates="drop")
        except Exception as e:
            print("Error in qcut:", e)
            bins = np.linspace(0, 1, 11)
        bin_labels = []
        for i in range(len(bins) - 1):
            if i == 0:
                bin_labels.append(f"[{round(bins[i],3)}, {round(bins[i+1],3)}]")
            else:
                bin_labels.append(f"({round(bins[i],3)}, {round(bins[i+1],3)}]")
        combined_df["similarity_bin"] = pd.cut(combined_df[sim_col],
                                            bins=bins,
                                            labels=bin_labels,
                                            include_lowest=True)
    else:
        bin_edges = np.linspace(0, 1, 11)
        bin_labels = []
        for i in range(len(bin_edges) - 1):
            if i == 0:
                bin_labels.append(f"[{round(bin_edges[i],1)}, {round(bin_edges[i+1],1)}]")
            else:
                bin_labels.append(f"({round(bin_edges[i],1)}, {round(bin_edges[i+1],1)}]")
        combined_df["similarity_bin"] = pd.cut(combined_df[sim_col],
                                               bins=bin_edges,
                                               labels=bin_labels,
                                               include_lowest=True)

    # Group the data by bin and leak_source.
    grouped = combined_df.groupby(["similarity_bin", "leak_source"])
    leaked_mean = grouped["leaked"].mean().reset_index()
    bin_counts = grouped.size().reset_index(name="count")
    merged_stats = pd.merge(leaked_mean, bin_counts, on=["similarity_bin", "leak_source"])
    merged_stats["leaked"] = merged_stats["leaked"].fillna(0)
    merged_stats["similarity_bin"] = pd.Categorical(merged_stats["similarity_bin"],
                                                    categories=bin_labels,
                                                    ordered=True)
    merged_stats.sort_values("similarity_bin", inplace=True)
    
    bin_ann = merged_stats[merged_stats["leak_source"]=="LiRA"].set_index("similarity_bin")["count"].to_dict()

    # -------------------------
    # Determine subplot title.
    # -------------------------
    parts = results_type.split("_")
    fp_part = "_".join(parts[2:])
    fp_name = fp_mapping.get(fp_part, fp_part)
    assay_name = assay_mapping.get(assay, assay)
    title_str = f"{assay_name} {fp_name}"
    
    # -------------------------
    # Plot grouped bar plot with hue.
    # -------------------------
    ax = axes[idx]
    hue_order = ["LiRA", "RMIA"]
    sns.barplot(
        x="similarity_bin",
        y="leaked",
        hue="leak_source",
        data=merged_stats,
        palette="muted",
        order=bin_labels,
        hue_order=hue_order,
        ax=ax
    )
    ax.set_title(title_str)

    # For the legend: display it only in the top-right subplot (idx == 2).
    if idx == 2:
        leg = ax.legend(title="Attack", loc="upper right")
        for text in leg.get_texts():
            text.set_fontsize(10)
    else:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # X-axis aesthetics.
    if idx >= 3:
        if similarity_mode == "average":
            ax.set_xlabel("Average Tanimoto similarity in training data")
            ax.tick_params(axis="x", labelsize=7, rotation=45)
        else:
            ax.set_xlabel("Highest Tanimoto similarity in training data")
            ax.tick_params(axis="x", labelsize=7)
    else:
        ax.set_xlabel("")
        ax.set_xticklabels([])


    # Y-axis aesthetics.
    if idx % 3 == 0:
        ax.set_ylabel("Fraction of molecules identified at FPR=0")
    else:
        ax.set_ylabel("")
    
    # Annotate each bin group once with the sample count.
    xticks = ax.get_xticks()  # one tick per bin
    for i, bin_label in enumerate(bin_labels):
        count = bin_ann.get(bin_label, 0)
        bin_data = merged_stats[merged_stats["similarity_bin"] == bin_label]
        if not bin_data.empty:
            max_height = bin_data["leaked"].max()
        else:
            max_height = 0
        ax.text(xticks[i], max_height, f"{count}", ha="center", va="bottom", fontsize=10)

plt.tight_layout()

# -------------------------
# Save the figure as a PDF.
# -------------------------
output_plot_dir = "..."
os.makedirs(output_plot_dir, exist_ok=True)
plot_filename = os.path.join(output_plot_dir, f"leakage_investigation_plot_{dataset}_{similarity_mode}.pdf")
plt.savefig(plot_filename, format="pdf")
print(f"Plot saved to {plot_filename}")

plt.show()

# -------------------------
# Save the statistical test results to a CSV file.
# -------------------------
output_dir = "..."
os.makedirs(output_dir, exist_ok=True)
stats_df = pd.DataFrame(stats_results)
output_file = os.path.join(output_dir, f"leakage_investigation_stats_{dataset}_{similarity_mode}.csv")
stats_df.to_csv(output_file, index=False)
print(f"Statistical test results saved to {output_file}")
