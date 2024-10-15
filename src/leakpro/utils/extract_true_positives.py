import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve
import pandas as pd
import torch
from rdkit import Chem
from moreno.utils.convert_representations import convert_dataset
from typing import Dict
import os

# Assuming `dataset` is your Pytorch dataset
# `scores` is a numpy array containing the model scores
# `labels` is a numpy array containing the true labels

def write_true_positives_to_disc(dataset: Dataset, scores: np.ndarray, labels: np.ndarray, mask: np.ndarray, configs: Dict, attack: str, higher_score_class_1: bool = True):

    # sanity check, turn off for performance
    compare_datasets(dataset, configs=configs, mask= mask)
    scores = scores.squeeze()

    original_scores = scores.copy()

    if not higher_score_class_1:
        scores = -scores
    
    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "thresholds": thresholds})
    # df.to_csv("/home/khcq385/LeakPro/data_dir/roc_curve.csv", index=False)

    # Find the threshold where FPR is 0
    threshold = thresholds[np.where(fpr == 0)[0][-1]]

    # Find indices of true positives at this threshold
    true_positive_indices = np.where((scores >= threshold) & (labels == 1))[0]

    # Extract the samples from the dataset
    true_positive_samples = [dataset[i][0] for i in true_positive_indices]

    # Convert the samples (which are tensors) to lists
    true_positive_samples = [sample.tolist() if isinstance(sample, torch.Tensor) else sample for sample in true_positive_samples]

    # Load the original data
    original_data_path = configs["train_data_path"]
    original_data = pd.read_csv(original_data_path)
    # filter out indices that are false in the mask
    mask = mask[:len(original_data)]
    original_data = original_data[mask]

    # Get rows of original data corresponding to true_positive_indices
    true_positive_originals = original_data.iloc[true_positive_indices]
    molecules = true_positive_originals["smiles"]
    property_label = true_positive_originals["label"]
    # Create a DataFrame with the relevant information
    sample_df = pd.DataFrame({
    #   "scores": original_scores[true_positive_indices], 
    #   "samples": true_positive_samples, 
        "molecules": molecules,
        "property_label": property_label
    })

    # Save the DataFrame to a CSV file
    output_dir = configs["report_log"] / "true_positives_at_FPR0"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{attack}.csv"
    sample_df.to_csv(output_path, index=False)


def compare_datasets(dataset: Dataset, configs: Dict, mask = None):
    train_csv_file = configs["train_data_path"]
    rep = configs["representation"]
    df = pd.read_csv(train_csv_file)
    if mask is not None:
        mask = mask[:len(df)]
        df = df[mask]
    file_dataset, _ = convert_dataset(df, rep)
    # check for the indices to make sure we didnt mess up the order
    for idx in range(len(file_dataset)):
        sample_original = file_dataset[idx]
        sample_input = dataset[idx]
        for o, i in zip(sample_original, sample_input):
            if isinstance(o, torch.Tensor) and isinstance(i, torch.Tensor):
                assert torch.allclose(o, i), f"Mismatch at index {idx}"
            elif isinstance(o, str) and isinstance(i, str):
                assert o == i, f"Mismatch at index {idx}"
            elif isinstance(o, list) and isinstance(i, list):
                # Ensure we only process non-empty lists
                while isinstance(o, list) and len(o) > 0:
                    o = o[0]
                while isinstance(i, list) and len(i) > 0:
                    i = i[0]
                if isinstance(o, Chem.Mol) and isinstance(i, Chem.Mol):
                    smiles1 = Chem.MolToSmiles(o)
                    smiles2 = Chem.MolToSmiles(i)
                    assert smiles1 == smiles2, f"Mismatch at index {idx}"
                else:
                    assert o == i, f"Mismatch at index {idx}"
            else:
                assert o == i, f"Mismatch at index {idx}"

        