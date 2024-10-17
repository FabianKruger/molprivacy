from moreno.data_modules.custom_datamodule import (
    CustomDataModule,
)
from moreno.config import Config
import pandas as pd
from typing import List, Optional
import torch
from rdkit import Chem
from moreno.utils.csv_file_operations import save_csv
from pathlib import Path


class UserDataModule(CustomDataModule):
    """Class for data management of Ames dataset. Desired representation of molecules must be specified in the initialization"""

    def __init__(
        self,
        representation: str,
        dataset_paths: List[str],
        split: List[float],
        batch_size: int = 64,
        num_workers: int = 8,
        prefetch_factor: int = 8,
    ):
        super().__init__(
            representation=representation,
            dataset_name="user_provided_dataset",
            split=split,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
        self.dataset_paths = dataset_paths
        self.train_data: Optional[pd.DataFrame] = None
        self.validation_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None

    def prepare_data(self) -> None:
        train_file = self.data_dir / "train.csv"
        validation_file = self.data_dir / "validation.csv"
        test_file = self.data_dir / "test.csv"
        if not (
            train_file.exists() and validation_file.exists() and test_file.exists()
        ):
            if len(self.dataset_paths) == 1:
                print(
                    "Found one dataset. This will be split into train val and test according to the split"
                )
                data = pd.read_csv(self.dataset_paths[0])
                self.sanity_check(data)
                data = data.sample(frac=1).reset_index(drop=True)
                self.train_data = data.iloc[: int(len(data) * self.split[0])]
                self.validation_data = data.iloc[
                    int(len(data) * self.split[0]) : int(
                        len(data) * (self.split[0] + self.split[1])
                    )
                ]
                self.test_data = data.iloc[
                    int(len(data) * (self.split[0] + self.split[1])) :
                ]
            # elif len(self.dataset_paths) == 2:
            #     print(f"Found 2 datasets. Using the first one ({self.dataset_paths[0]}) as training data and the second one {self.dataset_paths[1]}) as validation data.")
            #     train_data = pd.read_csv(self.dataset_paths[0])
            #     self.sanity_check(train_data)
            #     self.train_data = train_data
            #     validation_data = pd.read_csv(self.dataset_paths[1])
            #     self.sanity_check(validation_data)
            #     self.validation_data = validation_data
            elif len(self.dataset_paths) == 3:
                print(
                    f"Found 3 datasets. Using the first one ({self.dataset_paths[0]}) as training data, the second one ({self.dataset_paths[1]}) as validation data, and the last one ({self.dataset_paths[2]}) as test data."
                )
                train_data = pd.read_csv(self.dataset_paths[0])
                self.sanity_check(train_data)
                self.train_data = train_data
                validation_data = pd.read_csv(self.dataset_paths[1])
                self.sanity_check(validation_data)
                self.validation_data = validation_data
                test_data = pd.read_csv(self.dataset_paths[2])
                self.sanity_check(test_data)
                self.test_data = test_data
            else:
                raise ValueError(
                    f"Provided {len(self.dataset_paths)} dataset paths. Expected 1 or 3."
                )
            save_csv(self.train_data, "train.csv")
            save_csv(self.validation_data, "validation.csv")
            save_csv(self.test_data, "test.csv")

    def setup(self, stage: str) -> None:
        if stage == "fit":
            train_data = pd.read_csv(self.data_dir / "train.csv")
            total_negative_samples = len(train_data[train_data["label"] == 0])
            total_positive_samples = len(train_data[train_data["label"] == 1])
            Config.set_pos_weights(
                torch.tensor([total_negative_samples / total_positive_samples])
            )
            self.train_dataset = self.convert_dataset(train_data)
            validation_data = pd.read_csv(self.data_dir / "validation.csv")
            self.validation_dataset = self.convert_dataset(validation_data)
        if stage == "test":
            test_data = pd.read_csv(self.data_dir / "test.csv")
            self.test_dataset = self.convert_dataset(test_data)

    def sanity_check(self, data: pd.DataFrame) -> None:
        expected_columns = {"smiles", "label"}
        if set(data.columns) != expected_columns:
            raise ValueError(
                f"Column names of provided data are not as expected. Expected {expected_columns}, but got {set(data.columns)}."
            )
        if data.columns.value_counts().to_dict() != {
            col: 1 for col in expected_columns
        }:
            raise ValueError(
                f"Provided data had multiple columns for smiles or label. Expected exactly one of each {expected_columns}, but got {data.columns.tolist()}."
            )
        if len(data) < 10:
            raise ValueError(
                f"Found less than 10 molecules in the provided data. That's way too few. Amount is {len(data)}."
            )
        if not data["label"].isin([0, 1]).all():
            raise ValueError(
                f"The 'label' column contains values other than 0 and 1. This package only supports binary classification. Unique values found: {data['label'].unique()}."
            )
        invalid_smiles = data["smiles"].apply(lambda x: Chem.MolFromSmiles(x) is None)
        if invalid_smiles.any():
            invalid_entries = data["smiles"][invalid_smiles]
            raise ValueError(
                f"The 'smiles' column contains these invalid SMILES strings: {invalid_entries.tolist()}"
            )
