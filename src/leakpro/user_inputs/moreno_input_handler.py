
import logging

import torch
from torch import cuda, device, nn, optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, ConcatDataset
from tqdm import tqdm

from leakpro.user_inputs.abstract_input_handler import AbstractInputHandler

from moreno.utils.convert_representations import convert_dataset
from moreno.data_modules.datasets_and_collate_functions import (
    GraphDataset, CNNDataset, mol_collate_fn, CNN_collate_fn
)
from moreno.models.MLP import MLPLightning
from moreno.models.MPNN import MPNNLightning
from moreno.models.CNN import CNNLightning
import pandas as pd
import numpy as np
import yaml 
from typing import Optional, Dict, Any, Tuple, List, Type
import lightning as L 
from moreno.config import Config

class MorenoInputHandler(AbstractInputHandler):
    
    def __init__(self, configs: dict, logger:logging.Logger) -> None:
        super().__init__(configs = configs, logger = logger)

        self.hyperparameters: Dict[str, Any] = None
        self.input_vec_dim: Optional[int] = None
        self.inputs: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray]  = None
        self.lightning_model: Optional[L.LightningModule] = None
        self._target_model_blueprint: Optional[Type[L.LightningModule]] = None
        self.setup()

    def setup(self) -> None:
        """Set up the code handler by retrieving the model class, target metadata, trained target model, and population."""
        self._load_population()
        self._load_target_metadata()
        self._load_trained_target_model()
        

    def _load_population(self) -> None:
        train_data = pd.read_csv(self.configs["data"]["train_data_path"])
        test_data = pd.read_csv(self.configs["data"]["test_data_path"])
        self.logger.info(f"Training and test data loaded from {self.configs['data']['train_data_path']} and {self.configs['data']['test_data_path']}")

        train_dataset, self.input_vec_dim = convert_dataset(data= train_data, representation= self.configs["moreno"]["representation"])
        test_dataset, _ = convert_dataset(data = test_data, representation= self.configs["moreno"]["representation"])
        # will be initialzed properly later. Dont think the indexing works as intended otherwise with Subsets
        self.population = ConcatDataset([train_dataset, test_dataset])

        # Converting dataset to numpy arrays holding objects
        # Convert datasets to lists (necessary to make sure dtype is correct)
        train_inputs_list, train_labels_list = self._dataset_to_list(train_dataset)
        test_inputs_list, test_labels_list = self._dataset_to_list(test_dataset)

        # Explicitly create empty object arrays
        train_inputs = np.empty(len(train_inputs_list), dtype=object)
        train_labels = np.empty(len(train_labels_list), dtype=object)
        test_inputs = np.empty(len(test_inputs_list), dtype=object)
        test_labels = np.empty(len(test_labels_list), dtype=object)

        # Populate the object arrays with tensors
        for i, obj in enumerate(train_inputs_list):
            train_inputs[i] = obj

        for i, tensor in enumerate(train_labels_list):
            train_labels[i] = tensor

        for i, obj in enumerate(test_inputs_list):
            test_inputs[i] = obj

        for i, tensor in enumerate(test_labels_list):
            test_labels[i] = tensor


        # set the class imbalance
        num_pos = sum([(tensor == 1).sum().item() for tensor in test_labels])
        num_neg = sum([(tensor == 0).sum().item() for tensor in test_labels])
        Config.set_pos_weights(torch.tensor([(num_neg / num_pos)]))

        # Concatenate the numpy arrays
        self.inputs = np.concatenate((train_inputs, test_inputs), axis=0)
        self.labels = np.concatenate((train_labels, test_labels), axis=0)

        # Create population
        self.population = self.get_dataset(np.arange(len(train_data)+len(test_data)))

        # Create numpy arrays for the indices
        self._target_model_metadata["train_indices"] = np.arange(len(train_data))
        self._target_model_metadata["test_indices"] = np.arange(len(train_data), int(len(train_data) + 0.5*len(test_data)))
    
    def _load_model_class(self) -> None:
        pass

    def _load_target_metadata(self) -> None:
        hyperparms_path = self.configs["moreno"]["hyperparameters_path"]
        with open(hyperparms_path, 'r') as file:
            hyperparameters = yaml.safe_load(file)
        self.hyperparameters = hyperparameters
        self.logger.info(f"Loaded hyperparameters from {hyperparms_path}")

    def _validate_target_metadata(self) -> None:
        pass

    def _load_trained_target_model(self) -> None:
        model_path = self.configs["moreno"]["model_path"]
        if self.configs["moreno"]["representation"] in ["rdkit", "mold2", "transformer_vector", "ECFP4", "ECFP6", "MACCS", "custom"]:
            lightning_model = MLPLightning.load_from_checkpoint(model_path)
            self._target_model_blueprint = MLPLightning
        elif self.configs["moreno"]["representation"] == "graph":
            lightning_model = MPNNLightning.load_from_checkpoint(model_path)
            self._target_model_blueprint = MPNNLightning
        elif self.configs["moreno"]["representation"] == "transformer_matrix":
            lightning_model = CNNLightning.load_from_checkpoint(model_path)
            self._target_model_blueprint = CNNLightning
        else:
            raise ValueError(f"Unknown representation {self.configs['moreno']['representation']}")
        lightning_model.eval()
        self.target_model = lightning_model.model
        self.lightning_model = lightning_model

    def get_dataset(self, dataset_indices: np.ndarray) -> Dataset:
        # I assume that dataset_indices are just an array of numbers here and not an array of len(population) with true or false values
        self._validate_indices(dataset_indices)
        inputs_subset = self.inputs[dataset_indices]
        labels_subset = self.labels[dataset_indices]
        if self.configs["moreno"]["representation"] == "graph":
            input_list_of_lists = inputs_subset.tolist()
            dataset = GraphDataset(input_list_of_lists, labels_subset)
        elif self.configs["moreno"]["representation"] == "transformer_matrix":
            input_list = inputs_subset.tolist()
            dataset = CNNDataset(input_list, labels_subset)
        else:
            inputs_tensor = torch.stack(inputs_subset.tolist())
            labels_tensor = torch.stack(labels_subset.tolist())
            dataset = TensorDataset(inputs_tensor, labels_tensor)
        return dataset
    
    def get_dataloader(self, dataset_indices: np.ndarray, batch_size: int = 64) -> DataLoader:
        """Default implementation of the dataloader."""
        dataset = self.get_dataset(dataset_indices)
        if self.configs["moreno"]["representation"] == "graph":
            collate_fn = mol_collate_fn
        elif self.configs["moreno"]["representation"] == "transformer_matrix":
            collate_fn = CNN_collate_fn
        else:
            collate_fn = None
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)    
    
    def get_dataset_from_numpy_arrays(self, inputs: np.ndarray, labels: np.ndarray):
        if isinstance(inputs, (np.ndarray, torch.Tensor)):
            inputs = inputs.flatten()
            labels = labels.flatten()
        if self.configs["moreno"]["representation"] == "graph":
            if isinstance(inputs, (np.ndarray, torch.Tensor)):
                inputs = inputs.tolist()
            dataset = GraphDataset(inputs, labels)
        elif self.configs["moreno"]["representation"] == "transformer_matrix":
            if isinstance(inputs, (np.ndarray, torch.Tensor)):
                inputs = inputs.tolist()
            dataset = CNNDataset(inputs, labels)
        else:
            inputs_tensor = torch.stack(inputs.tolist())
            labels_tensor = torch.stack(labels.tolist())
            dataset = TensorDataset(inputs_tensor, labels_tensor)
        return dataset        
    
    def get_dataloader_from_dataset(self, dataset: Dataset):
        if self.configs["moreno"]["representation"] == "graph":
            collate_fn = mol_collate_fn
        elif self.configs["moreno"]["representation"] == "transformer_matrix":
            collate_fn = CNN_collate_fn
        else:
            collate_fn = None
        return DataLoader(dataset=dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)  
    
    def get_target_replica(self) -> Tuple[torch.nn.Module, nn.modules.loss._Loss, torch.optim.Optimizer]:
        # TODO: Probably wont work since it return a lightning Module instead
        # Try:
        # 1) return lightning_model.model caveat: probably wont work for the CNNLightning module. Try the following:
        # 2) restructure CNNLightning and create a nn.Module class that is called model in it
        replica_model_lightning = self._target_model_blueprint(**self.hyperparameters)
        
        return replica_model_lightning, None, None
    
    def get_criterion(self) -> torch.nn.modules.loss._Loss:
        """Get the loss function for the target model to be used in shadow model training."""
        return self.lightning_model.loss

    def get_optimizer(self, model:torch.nn.Module) -> torch.optim.Optimizer:
        """Get the optimizer used for the target model to be used in shadow model training."""
        return self.lightning_model.configure_optimizers()

    def train(
        self,
        dataloader: DataLoader,
        model: torch.nn.Module,
        criterion: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        epochs: int,
    ) -> nn.Module:
        """Procedure to train the shadow models on data from the population."""
        # TODO: add validation dataloader and early stopping

        trainer = L.Trainer(
                logger=False,
                enable_checkpointing=False,
                max_epochs= 15,
                accelerator="auto",
                enable_progress_bar=True,
            )
        
        trainer.fit(model, dataloader)
        model.eval()
        # return will not work for CNN, see above
        return {"model": model,
                "metrics": {"accuracy": None, "loss": None}}
    
    @property
    def population_size(self) -> int:
        """Get the size of the population."""
        return len(self.population)

    def _dataset_to_list(self, dataset: Dataset) -> Tuple[List[Any], List[Any]]:
        inputs_list: List[Any] = []
        labels_list: List[Any] = []

        for item in dataset:
            inputs, labels = item
            inputs_list.append(inputs)
            labels_list.append(labels)

        return inputs_list, labels_list