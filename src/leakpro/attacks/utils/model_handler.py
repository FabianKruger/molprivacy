"""Abstract class for the model handler."""

import joblib
import torch
from torch import load
from torch.nn import Module

from typing import Tuple
from leakpro.user_inputs.abstract_input_handler import AbstractInputHandler
from leakpro.utils.input_handler import (
    get_class_from_module,
    get_criterion_mapping,
    get_optimizer_mapping,
    import_module_from_file,
)


class ModelHandler:
    """Class to handle models used in attacks."""

    def __init__(
        self,
        handler: AbstractInputHandler,
    ) -> None:
        """Initialize the ModelHandler class."""
        self.logger = handler.logger
        self.handler = handler
        self.init_params = {}

    def _import_model_from_path(self, module_path: str, model_class: str) -> Module:
        """Import the model from the given path.

        Args:
        ----
            module_path (str): The path to the module.
            model_class (str): The name of the model class.

        Returns:
        -------
            Module: The imported blueprint of a model.

        """
        try:
            module = import_module_from_file(module_path)
            self.model_blueprint = get_class_from_module(module, model_class)
        except Exception as e:
            raise ValueError(
                f"Failed to create model blueprint from {model_class} in {module_path}"
            ) from e

    def _get_optimizer_class(self, optimizer_name: str) -> torch.optim.Optimizer:
        """Get the optimizer class based on the optimizer name.

        Args:
        ----
            optimizer_name (str): The name of the optimizer.

        Returns:
        -------
            torch.optim.Optimizer: The optimizer class.

        """
        try:
            self.optimizer_class = get_optimizer_mapping()[optimizer_name]
        except Exception as e:
            raise ValueError(
                f"Failed to create optimizer from {self.optimizer_config['name']}"
            ) from e

    def _get_criterion_class(self, criterion_name: str) -> torch.nn.Module:
        """Get the criterion class based on the criterion name.

        Args:
        ----
            criterion_name (str): The name of the criterion.

        Returns:
        -------
            torch.nn.Module: The criterion class.

        """
        try:
            self.criterion_class = get_criterion_mapping()[criterion_name]
        except Exception as e:
            raise ValueError(
                f"Failed to create criterion from {self.criterion_config['name']}"
            ) from e

    def _get_model_criterion_optimizer(self) -> Tuple[Module, Module, Module]:
        """Get the model, criterion, and optimizer from the handler or config."""

        # Set up shadow model from config file
        if self.model_blueprint is not None:
            model = self.model_blueprint(**self.init_params)
            optimizer = self.optimizer_class(
                model.parameters(), **self.optimizer_config
            )
            criterion = self.criterion_class(**self.loss_config)
        else:
            # Set up shadow model from handler
            model, criterion, optimizer = self.handler.get_target_replica()

        return model, criterion, optimizer

    def _load_model(self, model_path: str) -> Module:
        """Load a shadow model from a saved state.

        Args:
        ----
            model_path (str): The path to the saved model.

        Returns:
        -------
            Module: The loaded shadow model.

        """
        try:
            blueprint = (
                self.handler.target_model_blueprint
                if self.model_blueprint is None
                else self.model_blueprint
            )
            model = blueprint(**self.init_params)  # noqa: E501
            criterion = (
                self.handler.get_criterion()
                if self.criterion_class is None
                else self.criterion_class(**self.loss_config)
            )
        except Exception as e:
            raise ValueError("Failed to create model from blueprint") from e

        try:
            with open(model_path, "rb") as f:
                model.load_state_dict(load(f))
                self.logger.info(f"Loaded model from {model_path}")
            return model, criterion
        except FileNotFoundError as e:
            raise ValueError(f"Model file not found at {model_path}") from e

    def _load_metadata(self, metadata_path: str) -> dict:
        """Load metadata from a saved state.

        Args:
        ----
            metadata_path (str): The path to the saved metadata.

        Returns:
        -------
            dict: The loaded metadata.

        """
        try:
            with open(metadata_path, "rb") as f:
                return joblib.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Metadata at {metadata_path} not found") from e
