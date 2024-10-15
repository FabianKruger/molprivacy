from torch.utils.data import TensorDataset
from typing import Tuple
from moreno.data_modules.datasets_and_collate_functions import (
    GraphDataset, CNNDataset
)
from moreno_encoders.utils.encodings import generate_vector_encodings, generate_matrix_encodings
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
import numpy as np
import moreno.custom_representation


def convert_dataset(data: pd.DataFrame, representation: str) -> Tuple[TensorDataset | GraphDataset | CNNDataset, int]:

    labels = data["label"].to_numpy()
    input_vec_dim: int = 0

    if representation == "custom":
        if moreno.custom_representation.convert_vector is None:
            raise ValueError("Custom representation function was not found.")
        feature_array, input_vec_dim = moreno.custom_representation.convert_vector(data["smiles"].tolist()) # in: List of smiles, out: Tuple(Array of numpy arrays(vectors), input dimension in model)
        features_tensor = torch.tensor(feature_array, dtype=torch.float)
        labels_tensor = torch.tensor(labels, dtype=torch.float)
        dataset = TensorDataset(features_tensor, labels_tensor)
    elif representation == "ECFP4" or representation == "ECFP6":
        molecules = [Chem.MolFromSmiles(smiles) for smiles in data["smiles"]]
        if representation == "ECFP4":
            mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        else: 
            mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
        input_vec_dim = 2048
        features = np.array(
            [mfpgen.GetFingerprintAsNumPy(molecule) for molecule in molecules]
        )  # (N, 2048)
        features_tensor = torch.tensor(features, dtype=torch.float)
        labels_tensor = torch.tensor(labels, dtype=torch.float)
        dataset = TensorDataset(features_tensor, labels_tensor)
    elif representation == "MACCS":
        molecules = [Chem.MolFromSmiles(smiles) for smiles in data["smiles"]]
        maccs_keys = np.array([GetMACCSKeysFingerprint(mol) for mol in molecules])
        maccs_tensor = torch.tensor(maccs_keys, dtype=torch.float)
        labels_tensor = torch.tensor(labels, dtype=torch.float)
        input_vec_dim = 167
        dataset = TensorDataset(maccs_tensor, labels_tensor)
    elif representation == "rdkit":
        molecules = [Chem.MolFromSmiles(smiles) for smiles in data["smiles"]]
        rdkgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048)
        features = np.array(
            [rdkgen.GetFingerprintAsNumPy(molecule) for molecule in molecules]
        )  # (N, 2048)
        input_vec_dim = 2048
        features_tensor = torch.tensor(features, dtype=torch.float)
        labels_tensor = torch.tensor(labels, dtype=torch.float)
        dataset = TensorDataset(features_tensor, labels_tensor)            
    elif representation == "graph":
        molecules = [[Chem.MolFromSmiles(smiles)] for smiles in data["smiles"]] # chemprop wants mols as list of list with len(outer list) = number datapoints
        dataset = GraphDataset(molecules=molecules, labels=labels)
    elif representation == "transformer_vector":
        molecules = data["smiles"].to_list()
        encodings_tensor = generate_vector_encodings(molecules)
        input_vec_dim = encodings_tensor.shape[1] # 64
        labels_tensor = torch.tensor(labels, dtype=torch.float) # (N, 512)
        dataset = TensorDataset(encodings_tensor, labels_tensor)     
    elif representation == "transformer_matrix":
        molecules = data["smiles"].to_list()
        encodings_list = generate_matrix_encodings(molecules)
        dataset = CNNDataset(encodings_list, labels)
    else:
        raise NotImplementedError
    return dataset, input_vec_dim