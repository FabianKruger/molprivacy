import pandas as pd
from rdkit import Chem
from rdkit.Chem import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm import tqdm


def clean_smiles_df(raw_smiles: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Function to clean a pandas dataframe of smiles strings with corresponding labels.
    Specifically, it uses rdkits functions to sanitize, remove stereochemistry, remove salts, canonicalizes tautomers and gives the canonical smiles.
    In addition, dublicates and samples with different labels of the same smiles are removed.

    Args:
        raw_smiles (pd.DataFrame): Pandas dataframe that needs the columns "smiles" and "label".
        verbose (bool, optional): If True, prints every removed smiles and the reason for removal. Defaults to False.

    Returns:
        pd.DataFrame: Cleaned pandas dataframe.
    """
    raw_smiles.dropna(inplace=True)
    indices_to_drop = []
    for index, row in tqdm(raw_smiles.iterrows(), total=raw_smiles.shape[0]):
        smiles = row["smiles"]
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                if verbose:
                    print(f"Failed to parse SMILES: {smiles}")
                indices_to_drop.append(index)
                continue

            Chem.SanitizeMol(mol)
            mol = SaltRemover.SaltRemover().StripMol(mol, dontRemoveEverything=True)
            if len(Chem.GetMolFrags(mol)) > 1:
                if verbose:
                    print(
                        f"Multiple fragments for {smiles}. Dropped since label assignment is not clear."
                    )
                indices_to_drop.append(index)
                continue
            # this function has a bug in it and removes stereochemistry
            # mol = rdMolStandardize.CanonicalTautomer(mol)

            # make canonical smiles
            can_smiles = Chem.MolToSmiles(mol, canonical=True)
            if len(can_smiles)>200:
                if verbose:
                    print(f"The canonical version of the following smiles was too long: {smiles}")
                indices_to_drop.append(index)
                continue
            assert isinstance(
                can_smiles, str
            ), f"Canonical SMILES is not a string:{smiles} {can_smiles}"
            assert (
                can_smiles != ""
            ), f"Canonical SMILES is empty for smiles: {smiles} {can_smiles}"
            raw_smiles.at[index, "smiles"] = can_smiles
        except Exception as e:
            print(
                f"Failed to clean molecule: {smiles}. It will be dropped. Error: {str(e)}"
            )
            indices_to_drop.append(index)

    # Drop problematic rows
    print(f"Dropping {len(indices_to_drop)} rows.")
    raw_smiles.drop(indices_to_drop, inplace=True)

    # Sort the DataFrame, find dublicates, and filter them out (keeping the first occurence)
    raw_smiles = raw_smiles.sort_values(["smiles", "label"])
    duplicates = raw_smiles.duplicated(subset=["smiles", "label"], keep="first")
    raw_smiles = raw_smiles[~duplicates]

    # Check if there are any drugs with multiple labels
    drug_groups = raw_smiles.groupby("smiles")

    for smiles, group in drug_groups:
        if group["label"].nunique() > 1:
            if verbose:
                print(
                    f"Warning: Drug {smiles} has multiple different 'label' values. Smiles will be dropped."
                )
            raw_smiles = raw_smiles[raw_smiles["smiles"] != smiles]

    return raw_smiles
