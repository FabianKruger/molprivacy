from moreno.hyperparameter_optimizers.MLP_optimizer import (
    MLPOptimizer,
)
from moreno.hyperparameter_optimizers.MPNN_optimizer import (
    MPNNOptimizer,
)
from moreno.data_modules.custom_datamodule import (
    CustomDataModule,
)
from moreno.hyperparameter_optimizers.CNN_optimizer import (
    CNNOptimizer,
)
from moreno.hyperparameter_optimizers.optimizer import Optimizer
from moreno.models.MLP import MLPLightning
from moreno.models.MPNN import MPNNLightning
from moreno.models.CNN import CNNLightning
from typing import Optional


class ModelFactory:

    def __init__(
        self, datamodule: CustomDataModule, hyperparameter_optimization_time: int
    ) -> None:
        self.datamodule = datamodule
        self.hyperparameter_optimization_time = hyperparameter_optimization_time
        self.representation = datamodule.representation
        self.optimizer: Optional[Optimizer] = None

    def __str__(self) -> str:
        return f"Representation: {self.representation}, Hyperparameter optimization time: {self.hyperparameter_optimization_time}"

    def create_optimizer(self) -> None:
        # expandable with or statemet
        if self.representation in [
            "ECFP4",
            "ECFP6",
            "MACCS",
            "rdkit",
            "transformer_vector",
            "custom",
        ]:
            model_type = "MLP"
        elif self.representation == "graph":
            model_type = "MPNN"
        elif self.representation == "transformer_matrix":
            model_type = "CNN"
        else:
            raise NotImplementedError

        folder_name = "model"
        db_name = "optimization.db"
        training_log = "training"
        trained_model_name = "model"

        if model_type == "MLP":
            self.optimizer = MLPOptimizer(
                datamodule=self.datamodule,
                model_class=MLPLightning,
                result_folder_name=folder_name,
                result_db_name=db_name,
                training_log_name=training_log,
                trained_model_name=trained_model_name,
                optimization_time=self.hyperparameter_optimization_time,
            )
        elif model_type == "MPNN":
            self.optimizer = MPNNOptimizer(
                datamodule=self.datamodule,
                model_class=MPNNLightning,
                result_folder_name=folder_name,
                result_db_name=db_name,
                training_log_name=training_log,
                trained_model_name=trained_model_name,
                optimization_time=self.hyperparameter_optimization_time,
            )
        elif model_type == "CNN":
            self.optimizer = CNNOptimizer(
                datamodule=self.datamodule,
                model_class=CNNLightning,
                result_folder_name=folder_name,
                result_db_name=db_name,
                training_log_name=training_log,
                trained_model_name=trained_model_name,
                optimization_time=self.hyperparameter_optimization_time,
            )
        else:
            raise NotImplementedError

    def create_model(self):

        assert self.optimizer is not None
        self.optimizer.get_optimized_hyperparameters()
        self.optimizer.save_best_hyperparameter()
        self.optimizer.train_optimized_model()

    def test_model(self):
        assert self.optimizer is not None
        self.optimizer.test_model()
