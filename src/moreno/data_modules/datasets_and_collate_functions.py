from torch.utils.data import Dataset
from typing import List, Tuple
from rdkit.Chem import Mol
from numpy import ndarray
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence


class GraphDataset(Dataset):

    def __init__(self, molecules: List[List[Mol]], labels: ndarray):
        self.molecules = molecules
        self.labels = labels

    def __len__(self) -> int:
        return len(self.molecules)

    def __getitem__(self, idx: int) -> Tuple[List[Mol], ndarray]:
        return self.molecules[idx], self.labels[idx]


def mol_collate_fn(batch: List[Tuple[List[Mol], ndarray]]) -> Tuple[List[List[Mol]], torch.Tensor]:

    mols_batch = []
    labels_batch = []

    for mol, label in batch:
        mols_batch.append(mol)
        labels_batch.append(label)

    labels_tensor = torch.tensor(labels_batch, dtype=torch.float32)

    return mols_batch, labels_tensor

class CNNDataset(Dataset):
    def __init__(self, encodings: List[torch.Tensor], labels: np.ndarray):
        self.encodings = encodings
        if labels.dtype == np.object_:
            labels = np.array([label.item() for label in labels], dtype=np.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        encoding = self.encodings[idx]
        label = self.labels[idx]
        return encoding, label

def CNN_collate_fn(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    data = [item[0] for item in batch] # List[(S,E)]
    targets = torch.tensor([item[1] for item in batch]) # (B)
    data = pad_sequence(data, batch_first=True, padding_value=0) # (B, S, E)
    # If necessary, pad the sequences to have a sequence length of at least 20 to make sure all the filters will work
    if data.size(1) < 20:
        padding = torch.zeros((data.size(0), 20 - data.size(1), data.size(2)))
        data = torch.cat((data, padding), dim=1)
    return data, targets