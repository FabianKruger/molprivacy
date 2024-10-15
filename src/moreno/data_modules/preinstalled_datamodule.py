from moreno.utils.csv_file_operations import save_csv
from moreno.data_modules.custom_datamodule import (
    CustomDataModule,
)
from moreno.config import Config
import pandas as pd
from pathlib import Path
from typing import List
import torch

class PreInstalledDataModule(CustomDataModule):
    """Class for data management of pre-installed dataset. Desired representation of molecules must be specified in the initialization"""

    def __init__(
        self,
        representation: str,
        dataset_name: str,
        split: List[float],
        batch_size: int = 64,
        num_workers: int = 8,
        prefetch_factor: int = 8,
    ):
        super().__init__(
            representation=representation,
            dataset_name=dataset_name,
            split=split,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )

    def prepare_data(self) -> None:
        train_file = self.data_dir / "train.csv"
        validation_file = self.data_dir / "validation.csv"
        test_file = self.data_dir / "test.csv"
        if not (
            train_file.exists() and validation_file.exists() and test_file.exists()
        ):
            try:
                file_path = Config.get_package_data_dir() / f"{self.dataset_name}.csv"
                
                # Check if the file exists before trying to read it
                if not file_path.exists():
                    raise FileNotFoundError(f"No dataset found at {self.data_dir / f'{self.dataset_name}.csv'}. Try to run 'python -m moreno download' first.")
                data = pd.read_csv(file_path)

            except pd.errors.EmptyDataError:
                raise ValueError(f"The dataset at {file_path} is empty. Please check the file or download a valid dataset.")
                
            except pd.errors.ParserError:
                raise ValueError(f"The dataset at {file_path} is malformed or not in CSV format. Please verify the contents of the file.")
                
            except PermissionError:
                raise PermissionError(f"Permission denied: Unable to access the dataset at {file_path}. Check your file permissions.")
                
            except Exception as e:
                raise RuntimeError(f"An unexpected error occurred while trying to read the dataset at {file_path}: {e}. Please check the file and try again.")
            
            data = data.sample(frac=1).reset_index(drop=True) # shuffle
            train_data = data.iloc[: int(len(data) * self.split[0])]
            validation_data = data.iloc[int(len(data) * self.split[0]) : int(len(data) * (self.split[0] + self.split[1]))]
            test_data = data.iloc[int(len(data) * (self.split[0] + self.split[1])) : ]
            save_csv(train_data, "train.csv")
            save_csv(validation_data, "validation.csv")
            save_csv(test_data, "test.csv")

    def setup(self, stage: str) -> None:
        if stage == "fit":
            train_data = pd.read_csv(self.data_dir / "train.csv")
            total_negative_samples = len(train_data[train_data["label"] == 0])
            total_positive_samples = len(train_data[train_data["label"] == 1])
            Config.set_pos_weights(torch.tensor([total_negative_samples/total_positive_samples]))
            self.train_dataset = self.convert_dataset(train_data)
            validation_data = pd.read_csv(self.data_dir / "validation.csv")
            self.validation_dataset = self.convert_dataset(validation_data)
        if stage == "test":
            test_data = pd.read_csv(self.data_dir / "test.csv")
            self.test_dataset = self.convert_dataset(test_data)

