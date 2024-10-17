# constructs lightning datamodule
from moreno.data_modules.user_datamodule import UserDataModule
from moreno.data_modules.preinstalled_datamodule import PreInstalledDataModule
from moreno.data_modules.custom_datamodule import (
    CustomDataModule,
)
from typing import List, Optional


class DataFactory:

    def __init__(
        self,
        dataset: str,
        split: List[float],
        representation: str,
        paths: Optional[List[str]] = None,
    ) -> None:
        self.dataset = dataset
        self.split = split
        self.representation = representation
        self.paths = paths

    def __str__(self) -> str:
        return f"Dataset: {self.dataset}, Split: {self.split}, Representation: {self.representation}"

    def create_datamodule(self) -> CustomDataModule:
        datamodule = None
        if self.dataset in ["bbb", "ames", "del", "herg"]:
            datamodule = PreInstalledDataModule(
                representation=self.representation,
                split=self.split,
                dataset_name=self.dataset,
            )
        elif self.dataset == "file":
            assert self.paths is not None
            datamodule = UserDataModule(
                representation=self.representation,
                split=self.split,
                dataset_paths=self.paths,
            )
        else:
            raise NotImplementedError(
                f"Dataset {self.dataset} is not supported by the datafactory yet."
            )

        datamodule.prepare_data()
        datamodule.setup(stage="fit")
        return datamodule
