import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from moreno.hyperparameter_optimizers.optimizer import Optimizer
from moreno.models.MPNN import MPNNLightning
from moreno.data_modules.custom_datamodule import (
    CustomDataModule,
)
from typing import Type, Optional, Literal, cast


class MPNNOptimizer(Optimizer[MPNNLightning]):
    """
    Class for finding the best hyperparameters for an MPNN and training a model with them.
    """

    def __init__(
        self,
        datamodule: CustomDataModule,
        model_class: Type[MPNNLightning],
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

    def objective(self, trial: optuna.trial.Trial) -> float:

        # hyperparameters (taking the ranges from chemprop when possible and reasonable)
        message_passing_steps = trial.suggest_int("message_passing_steps", 2, 6)
        encoder_dropout = trial.suggest_uniform("encoder_dropout", 0.0, 0.4)
        encoder_hidden_size = trial.suggest_int("encoder_hidden_size", 300, 2400)
        add_bias_to_encoder_layers = trial.suggest_categorical(
            "add_bias_to_encoder_layers", [True, False]
        )
        aggregation: Literal["mean", "sum", "norm"] = cast(
            Literal["mean", "sum", "norm"],
            trial.suggest_categorical("aggregation", ["mean", "sum", "norm"]),
        )
        classifier_num_hidden_layers = trial.suggest_int(
            "classifier_num_hidden_layers", 1, 3
        )
        classifier_hidden_dim = trial.suggest_int("classifier_hidden_dim", 300, 2400)
        classifier_dropout = trial.suggest_uniform("classifier_dropout", 0.0, 0.4)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=False)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=False)

        # instantiate model
        model = self.model_class(
            message_passing_steps=message_passing_steps,
            encoder_dropout=encoder_dropout,
            encoder_hidden_size=encoder_hidden_size,
            add_bias_to_encoder_layers=add_bias_to_encoder_layers,
            aggregation=aggregation,
            classifier_num_hidden_layers=classifier_num_hidden_layers,
            classifier_hidden_dim=classifier_hidden_dim,
            classifier_dropout=classifier_dropout,
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

        trainer.fit(model, datamodule=self.datamodule)

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
            message_passing_steps=self.optimized_hyperparameters[
                "message_passing_steps"
            ],
            encoder_dropout=self.optimized_hyperparameters["encoder_dropout"],
            encoder_hidden_size=self.optimized_hyperparameters["encoder_hidden_size"],
            add_bias_to_encoder_layers=self.optimized_hyperparameters[
                "add_bias_to_encoder_layers"
            ],
            aggregation=self.optimized_hyperparameters["aggregation"],
            classifier_num_hidden_layers=self.optimized_hyperparameters[
                "classifier_num_hidden_layers"
            ],
            classifier_hidden_dim=self.optimized_hyperparameters[
                "classifier_hidden_dim"
            ],
            classifier_dropout=self.optimized_hyperparameters["classifier_dropout"],
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

        trainer.fit(model=final_model, datamodule=self.datamodule)
