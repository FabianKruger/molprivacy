import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdCoordGen
from rdkit.Chem import rdDepictor
import pandas as pd
import math

if __name__ == "__main__":
    # Read the CSV file
    df = pd.read_csv(".../privacy/results/true_positives_at_FPR0/lira.csv")

    # Sort the dataframe by the 'property_label' column
    df = df.sort_values(by="property_label")

    # Create molecules from SMILES strings and compute 2D coordinates
    mols = []
    for i, smiles in enumerate(df["molecules"]):
        mol = Chem.MolFromSmiles(smiles)
        if i == 0:
            # Use rdDepictor with ring templates for the first molecule
            # rdDepictor.Compute2DCoords(mol, useRingTemplates=True)
            pass
        else:
            # Use rdCoordGen for the rest of the molecules
            rdCoordGen.AddCoords(mol)
        mols.append(mol)

    # Calculate grid dimensions
    n_mols = len(mols)
    n_cols = min(7, n_mols)  # Maximum 7 columns
    n_rows = math.ceil(n_mols / n_cols)

    # Create a figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    # Flatten axes array for easier indexing
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

    # Plot molecules and labels
    for i, mol in enumerate(mols):
        if i < len(axes):
            img = Draw.MolToImage(mol, size=(300, 300))
            axes[i].imshow(img)
            axes[i].axis("off")

            # Add label below the image
            label = df["property_label"].iloc[i]
            if label == 0:
                axes[i].text(
                    0.5,
                    -0.1,
                    f"Label: {label}",
                    ha="center",
                    va="top",
                    transform=axes[i].transAxes,
                    fontsize=20,
                    fontweight="bold",
                    bbox=dict(
                        facecolor="white", edgecolor="black", boxstyle="round,pad=0.2"
                    ),
                )
            else:
                axes[i].text(
                    0.5,
                    -0.1,
                    f"Label: {label}",
                    ha="center",
                    va="top",
                    transform=axes[i].transAxes,
                    fontsize=20,
                )

    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout and display the figure
    plt.tight_layout()
    plt.savefig(".../leaked_molecules.pdf", bbox_inches="tight")
