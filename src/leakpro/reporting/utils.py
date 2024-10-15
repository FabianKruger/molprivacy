"""Utility functions for generating privacy risk report."""

from typing import Dict
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc

def prepare_priavcy_risk_report(
    audit_results: Dict,
    configs: dict,
    save_path: str = None,
) -> None:
    """Generate privacy risk report based on the auditing report.

    Args:
    ----
        audit_results(List): Privacy meter results.
        configs (dict): Auditing configuration.
        save_path (str, optional): Report path. Defaults to None.

    Raises:
    ------
        NotImplementedError: Check if the report for the privacy game is implemented.

    """
    if save_path is None:
        raise ValueError("Please provide a save path for the report")

    if audit_results is None:
        raise ValueError("Please provide the audit results")

    # Generate privacy risk report for auditing the model

    fpr = audit_results["fpr"]
    tpr = audit_results["tpr"]
    assert len(fpr) == len(tpr), "Something went wrong with the roc curve generation. FPR and TPR are not the same length."
    roc_csv = pd.DataFrame({'fpr': fpr, 'tpr': tpr})

    path = Path(save_path)
    path.mkdir(parents=True, exist_ok=True)
    roc_csv.to_csv((path / "ROC.csv"), index=False)

    # print the roc curve TODO: remove top spine
    roc_auc = auc(fpr, tpr)  
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))    
    # Find the minimum non-zero FPR value
    min_fpr = np.min(fpr[fpr > 0])
    plt.loglog(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})', color='blue', lw=2)
    plt.loglog([min_fpr, 1], [min_fpr, 1], linestyle='--', color='gray', lw=2)
    plt.xlim([min_fpr, 1.0])
    plt.ylim([min_fpr, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Training Data Identification', fontsize=14)
    plt.legend(loc='lower right')
    plt.gca().spines['top'].set_visible(False)
    plt.savefig((path / "privacy_performance.pdf"), format="pdf")
    plt.close()

    # print result txt with tpr for fpr = 0, fpr=10-3 and auroc
    # Highest TPR value at FPR = 0
    tpr_at_zero_fpr = tpr[fpr == 0].max()
    # Highest TPR value at FPR closest to 1e-3
    closest_fpr = fpr[np.argmin(np.abs(fpr - 1e-3))]
    tpr_at_1e3_fpr = tpr[fpr == closest_fpr].max()
    with open(path / "privacy_performance_overview.txt", "w") as f:
        f.write(f"Privacy performance in terms of identification of training data samples.\n")
        f.write(f"AUROC: {roc_auc:.4f}\n")
        f.write(f"TPR at FPR=0: {tpr_at_zero_fpr:.4f}\n")
        f.write(f"TPR at FPR closest to 1e-3: {tpr_at_1e3_fpr:.4f} (actual FPR: {closest_fpr:.5f})\n")
