import lightning as L
import optuna
from moreno.data_modules.custom_datamodule import (
    CustomDataModule,
)
from pathlib import Path
from typing import Optional, Type, TypeVar, Generic
import yaml
from abc import ABC, abstractmethod
from moreno.config import Config
import torch
from moreno.utils.plot_roc import plot_roc
import pandas as pd

T = TypeVar("T", bound=L.LightningModule)


class Optimizer(Generic[T], ABC):
    """
    Class for finding the best hyperparameters for a model class and training a model with them.
    """

    def __init__(
        self,
        datamodule: CustomDataModule,
        model_class: Type[T],
        result_folder_name: str,
        result_db_name: str,
        training_log_name: str,
        trained_model_name: str,
        optimization_time: int,
        optimized_hyperparameters: Optional[dict] = None,
        percentage_of_validation_set_to_use: float = 0.5,
        max_epochs: int = 20,
    ) -> None:
        """Initiating the optimizer

        Args:
            datamodule (CustomDataModule): A lightning datamodule with attribute self.input_vec_dim specifying the vector length of the inputs.
            model_class (L.LightningModule): The model class which should be optimized.
            result_folder_name (str): Created subfolder name of the cwd which will be created and contains the results.
            result_db_name (str): Results of the optimization process.
            training_log_name (str): Log of the training of the final optimized mode.l
            trained_model_name (str): Name of the final optimized and trained model.
            optimization_time (int): Time in seconds to run the hyperparameter search.
            optimized_hyperparameters (Optional[dict], optional): If optimized hyperparameters already have been searched for, they can be provided here. Skips the searching. Defaults to None.
            percentage_of_validation_set_to_use (float, optional): How much of the validation set should be used when searching for optimized hyperparameters. Defaults to 0.5.
            max_epochs (int, optional): Maximum number of epochs in each hyperparameter search. Defaults to 50.
        """
        self.model_class = model_class
        self.datamodule = datamodule
        self.result_folder_name = result_folder_name
        self.result_db_name = result_db_name
        self.training_log_name = training_log_name
        self.trained_model_name = trained_model_name
        self.optimization_time = optimization_time
        self.optimized_hyperparameters = optimized_hyperparameters
        # some general parameters
        self.percentage_of_validation_set_to_use = percentage_of_validation_set_to_use
        self.max_epochs = max_epochs
        # create result_folder and save path
        directory = Config.get_directory()
        self.subfolder_path = directory / self.result_folder_name
        self.subfolder_path.mkdir(exist_ok=True)

    @abstractmethod
    def objective(self, trial: optuna.trial.Trial) -> float:
        """Optunas objective function for hyperparameter evaluation.

        Args:
            trial (optuna.trial.Trial): Optuna Trial object

        Returns:
            float: Performance of the hyperparameters.
        """
        pass

    @abstractmethod
    def get_optimized_hyperparameters(self) -> None:
        """Search for optimized hyperparameters. Saves search to database."""
        pass

    @abstractmethod
    def train_optimized_model(self) -> None:
        """Uses the optimized hyperparameter attribute to create and train a model and save it. Also saves the training log."""
        pass

    def save_best_hyperparameter(self) -> None:
        """Save best hyperparameters in result folder."""
        file_path = self.subfolder_path / "optimized_hyperparameters.yaml"
        with open(file_path, "w") as yaml_file:
            yaml.dump(self.optimized_hyperparameters, yaml_file, default_flow_style=False)

    def test_model(self) -> None:
        """Test the trained model with the test set."""
        trained_model_path = self.subfolder_path / "model.ckpt"
        model = self.model_class.load_from_checkpoint(trained_model_path)
        model.eval()
        self.datamodule.setup(stage="test")
        test_dataloader = self.datamodule.test_dataloader()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for x, y in test_dataloader:
                if isinstance(x, torch.Tensor):
                    logits = model(x.to(model.device))
                else: 
                    logits = model(x)
                predictions = torch.sigmoid(logits.squeeze())
                if predictions.dim() == 0:
                    predictions = predictions.unsqueeze(0)
                assert predictions.shape == y.shape, "Predictions and labels dont have the same shape."
                for pred, label in zip(predictions, y):
                    y_pred.extend(pred.detach().flatten().tolist())
                    y_true.extend(label.detach().flatten().tolist())
        plot_roc(y_pred=y_pred, y_true=y_true, save_location=self.subfolder_path)
        roc = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
        roc.to_csv(self.subfolder_path / "roc.csv", index=False)
