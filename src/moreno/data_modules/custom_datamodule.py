import lightning as L
from moreno.config import Config
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, List
from abc import ABC, abstractmethod
from moreno.data_modules.datasets_and_collate_functions import (
    mol_collate_fn, GraphDataset, CNN_collate_fn, CNNDataset
)
from moreno.utils.convert_representations import convert_dataset
import pandas as pd


class CustomDataModule(L.LightningDataModule, ABC):

    def __init__(
        self,
        representation: str,
        dataset_name: str,
        split: List[float],
        batch_size: int = 64,
        num_workers: int = 8,
        prefetch_factor: int = 8,
    ) -> None:
        super().__init__()
        self.representation = representation
        self.dataset_name = dataset_name
        self.data_dir = Config.get_data_dir()
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.train_dataset: Optional[TensorDataset | GraphDataset | CNNDataset] = None
        self.validation_dataset: Optional[TensorDataset | GraphDataset | CNNDataset] = None
        self.test_dataset: Optional[TensorDataset | GraphDataset | CNNDataset] = None
        self.input_vec_dim: int = 0

    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod
    def setup(self, stage: str):
        pass

    def train_dataloader(self) -> DataLoader:
        if self.representation == "graph":
            collate_function = mol_collate_fn
        elif self.representation == "transformer_matrix":
            collate_function = CNN_collate_fn
        else:
            collate_function = None
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_function,
        )

    def val_dataloader(self) -> DataLoader:
        if self.representation == "graph":
            collate_function = mol_collate_fn
        elif self.representation == "transformer_matrix":
            collate_function = CNN_collate_fn
        else:
            collate_function = None
        assert self.validation_dataset is not None
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_function,
        )

    def test_dataloader(self) -> DataLoader:
        if self.representation == "graph":
            collate_function = mol_collate_fn
        elif self.representation == "transformer_matrix":
            collate_function = CNN_collate_fn
        else:
            collate_function = None
        assert self.test_dataset is not None
        # # to allow for random subsampling of the test dataset
        # sampler = RandomSampler(self.test_dataset, replacement=True, num_samples=len(self.test_dataset))
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_function,
            # sampler = sampler
        )

    def convert_dataset(self, data: pd.DataFrame) -> TensorDataset | GraphDataset | CNNDataset:
        """Converting a pandas dataframe to a pytorch tensor dataset with the smiles represented as what is specified in self.representation

        Args:
            data (pd.DataFrame): Dataframe with smiles and label columns

        Raises:
            NotImplementedError: Transforming smiles to representation specified in self.representation is not implemented

        Returns:
            Dataset: pytorch dataset with smiles represented as what is specified in self.representation
        """
        dataset, input_vec_dim = convert_dataset(data=data, representation= self.representation)
        self.input_vec_dim = input_vec_dim
        return dataset