import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from moreno.hyperparameter_optimizers.optimizer import Optimizer
from moreno.models.MLP import MLPLightning
from moreno.data_modules.custom_datamodule import (
    CustomDataModule,
)
from typing import Type, Optional


class MLPOptimizer(Optimizer[MLPLightning]):
    """
    Class for finding the best hyperparameters for an MLP and training a model with them.
    """

    def __init__(
        self,
        datamodule: CustomDataModule,
        model_class: Type[MLPLightning],
        result_folder_name: str,
        result_db_name: str,
        training_log_name: str,
        trained_model_name: str,
        optimization_time: int,
        optimized_hyperparameters: Optional[dict] = None,
        percentage_of_validation_set_to_use: float = 1.0,
        max_epochs: int = 20,
    ) -> None:
        super().__init__(
            datamodule=datamodule,
            model_class=model_class,
            result_folder_name=result_folder_name,
            result_db_name=result_db_name,
            training_log_name=training_log_name,
            trained_model_name=trained_model_name,
            optimization_time=optimization_time,
            optimized_hyperparameters=optimized_hyperparameters,
            percentage_of_validation_set_to_use=percentage_of_validation_set_to_use,
            max_epochs=max_epochs,
        )

        # creates dataloaders here, so they don't need to be created everytime in the objective function.
        self.datamodule.prepare_data()
        self.datamodule.setup(stage="fit")
        self.train_loader = self.datamodule.train_dataloader()
        self.val_loader = self.datamodule.val_dataloader()

    def objective(self, trial: optuna.trial.Trial) -> float:
        """Optunas objective function for hyperparameter evaluation.

        Args:
            trial (optuna.trial.Trial): Optuna Trial object

        Returns:
            float: Performance of the hyperparameters.
        """

        # hyperparameters
        dropout = trial.suggest_uniform("dropout", 0.0, 1.0)
        hidden_dim = trial.suggest_int("hidden_dim", 10, 1000)
        num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 5)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=False)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=False)

        # instantiate model
        model = self.model_class(
            dropout=dropout,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            input_vec_dim=self.datamodule.input_vec_dim,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        # make pruning callback
        callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")
        early_stopping_callback = EarlyStopping(monitor="val_loss", patience=2)

        # instantiate trainer
        trainer = L.Trainer(
            logger=False,
            limit_val_batches=self.percentage_of_validation_set_to_use,
            enable_checkpointing=False,
            max_epochs=self.max_epochs,
            accelerator="auto",
            callbacks=[callback, early_stopping_callback],
            enable_progress_bar=False,
        )

        # in case of logging for debugging
        # hyperparameters = dict(
        #     dropout=dropout,
        #     hidden_dim=hidden_dim,
        #     num_hidden_layers=num_hidden_layers,
        #     learning_rate=learning_rate,
        #     weight_decay=weight_decay,
        # )

        # assert trainer.logger is not None
        # trainer.logger.log_hyperparams(hyperparameters)

        trainer.fit(model, self.train_loader, self.val_loader)

        callback.check_pruned()

        return trainer.callback_metrics["val_loss"].item()

    def get_optimized_hyperparameters(self) -> None:
        """Search for optimized hyperparameters. Saves search to database."""

        if self.optimized_hyperparameters is not None:
            pass

        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=15,
        )

        # make subfolder to save results in and create paths

        db_file_path = self.subfolder_path / self.result_db_name
        storage_string = f"sqlite:///{db_file_path}"

        study = optuna.create_study(
            direction="minimize",
            pruner=pruner,
            storage=storage_string,
        )
        study.optimize(self.objective, n_trials=10000, timeout=self.optimization_time)

        self.optimized_hyperparameters = study.best_params

    def train_optimized_model(self) -> None:
        """Uses the optimized hyperparameter attribute to create and train a model and save it. Also saves the training log."""
        # instantiate model
        assert self.optimized_hyperparameters is not None

        final_model = self.model_class(
            dropout=self.optimized_hyperparameters["dropout"],
            hidden_dim=self.optimized_hyperparameters["hidden_dim"],
            num_hidden_layers=self.optimized_hyperparameters["num_hidden_layers"],
            input_vec_dim=self.datamodule.input_vec_dim,
            learning_rate=self.optimized_hyperparameters["learning_rate"],
            weight_decay=self.optimized_hyperparameters["weight_decay"],
        )

        # create logger
        logger = CSVLogger(self.subfolder_path, name=self.training_log_name)

        # create callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.subfolder_path,
            filename=self.trained_model_name,
            monitor="val_loss",
        )
        early_stopping_callback = EarlyStopping(monitor="val_loss", patience=10)

        # instantiate trainer
        trainer = L.Trainer(
            logger=logger,
            enable_checkpointing=True,
            accelerator="auto",
            callbacks=[checkpoint_callback, early_stopping_callback],
            enable_progress_bar=False,
        )

        trainer.fit(final_model, self.train_loader, self.val_loader)
