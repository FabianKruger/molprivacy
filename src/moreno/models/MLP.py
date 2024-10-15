# simple MLP model with Pytorch lightning and Optuna integration. For now just written for binary classification.
# Possible improvements: Add multiclass support, add reduce lr on plateau, add different optimizers as hyperparameter.
################################################################

import lightning as L
import torch
from torch import nn
from typing import List, Any, Tuple, Dict
from moreno.config import Config


class Model(nn.Module):
    def __init__(
        self,
        dropout: float,
        hidden_dim: int,
        num_hidden_layers: int,
        input_vec_dim: int,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_vec_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_vec_dim = hidden_dim
        layers.append(nn.ReLU())
        layers.append(nn.Linear(input_vec_dim, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.layers(x)
        return logits


class MLPLightning(L.LightningModule):
    def __init__(
        self,
        dropout: float,
        hidden_dim: int,
        num_hidden_layers: int,
        input_vec_dim: int,
        learning_rate: float,
        weight_decay: float = 1e-4,
    ) -> None:

        super().__init__()

        self.model = Model(
            dropout=dropout,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            input_vec_dim=input_vec_dim,
        )
        # saving all hyperparameters for easy access later
        self.dropout_rate = dropout
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.input_vec_dim = input_vec_dim
        self.lr = learning_rate
        self.weight_decay = weight_decay

        # use BCE loss with logits
        self.loss = nn.BCEWithLogitsLoss(pos_weight=Config.get_pos_weights())

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return logits

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        if logits.dim() > 1:
            logits = logits.squeeze()
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)
        loss = self.loss(logits, y)
        # log loss
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], *args: Any, **kwargs: Any
    ) -> None:
        x, y = batch
        logits = self(x)
        if logits.dim() > 1:
            logits = logits.squeeze()
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)
        loss = self.loss(logits, y)
        self.log("val_loss", loss)
        predictions = torch.sigmoid(logits)
        accuracy = predictions.round().eq(y.round()).float().mean()
        self.log("val_accuracy", accuracy)

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], *args: Any, **kwargs: Any
    ) -> None:
        x, y = batch
        logits = self(x)
        if logits.dim() > 1:
            logits = logits.squeeze()
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)
        loss = self.loss(logits, y)
        self.log("test_loss", loss)
        predictions = torch.sigmoid(logits)
        accuracy = predictions.round().eq(y.round()).float().mean()
        self.log("test_accuracy", accuracy)


    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer
