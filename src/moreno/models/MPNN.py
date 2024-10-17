import torch.nn as nn
import torch
import lightning as L
from chemprop.args import TrainArgs
from chemprop.models.mpn import MPN
from chemprop.nn_utils import initialize_weights
from typing import Literal, List, Tuple, Any, Dict
from rdkit.Chem import Mol
from moreno.config import Config


class MPNN(nn.Module):
    """Message passing neural network based on the chemprop implementation."""

    def __init__(
        self,
        message_passing_steps: int = 3,
        encoder_dropout: float = 0.0,
        encoder_hidden_size: int = 300,
        add_bias_to_encoder_layers: bool = False,
        aggregation: Literal["mean", "sum", "norm"] = "mean",
        classifier_num_hidden_layers: int = 3,
        classifier_hidden_dim: int = 200,
        classifier_dropout: float = 0.0,
    ):
        """Initialize the MPNN.

        Args:
            message_passing_steps (int, optional): Number of steps used to update the hidden vector before aggregating it in the atomic embedding. Defaults to 3.
            encoder_dropout (float, optional): Dropout for the NN's of the encoder. Defaults to 0.0.
            encoder_hidden_size (int, optional): Hidden layer size of the NN that maps concatenated bond and (neighboring) atomic vector to the hidden vector that is then used to update the atomic vector. Defaults to 300.
            add_bias_to_encoder_layers (bool, optional): Adds bias to the encoder NN layers. Defaults to False.
            aggregation (Literal["mean", "sum", "norm"], optional): How to combine the atomic representations to a molecule representation. Defaults to "mean".
            classifier_num_hidden_layers (int, optional): Number of hidden layers in the classifier. Defaults to 3.
            classifier_hidden_dim (int, optional): Hidden dimension of the classifier NN. Defaults to 200.
            classifier_dropout (float, optional): Dropout for the classifier layers. Defaults to 0.0.
        """

        super().__init__()

        args = TrainArgs()
        args.activation = "ReLU"  # don't want to change this for now, possible: 'ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'
        args.depth = message_passing_steps  # number of steps used to update the hidden vector before aggregating it in the atmoic embedding
        args.dropout = encoder_dropout
        args.hidden_size = encoder_hidden_size  # hidden layer size of the NN that maps concatenated bond and (neighboring) atomic vector to the hidden vector that is then used to update the atomic vector
        args.bias = add_bias_to_encoder_layers
        args.aggregation = aggregation

        # Uses message passing network from chemprop
        self.encoder = MPN(args)

        input_vec_dim = (
            args.hidden_size
        )  # output of MPN will be a vector of this dimension

        layers: List[nn.Module] = []
        for _ in range(classifier_num_hidden_layers):
            layers.append(nn.Linear(input_vec_dim, classifier_hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(classifier_dropout))
            input_vec_dim = classifier_hidden_dim
        layers.append(nn.ReLU())
        layers.append(nn.Linear(input_vec_dim, 1))
        self.classifier = nn.Sequential(*layers)

    def forward(self, molecules: List[List[Mol]]) -> torch.Tensor:
        encoding = self.encoder(molecules)
        logits = self.classifier(encoding)
        return logits


class MPNNLightning(L.LightningModule):

    def __init__(
        self,
        message_passing_steps: int = 3,
        encoder_dropout: float = 0.0,
        encoder_hidden_size: int = 300,
        add_bias_to_encoder_layers: bool = False,
        aggregation: Literal["mean", "sum", "norm"] = "mean",
        classifier_num_hidden_layers: int = 3,
        classifier_hidden_dim: int = 200,
        classifier_dropout: float = 0.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()

        self.model = MPNN(
            message_passing_steps=message_passing_steps,
            encoder_dropout=encoder_dropout,
            encoder_hidden_size=encoder_hidden_size,
            add_bias_to_encoder_layers=add_bias_to_encoder_layers,
            aggregation=aggregation,
            classifier_num_hidden_layers=classifier_num_hidden_layers,
            classifier_hidden_dim=classifier_hidden_dim,
            classifier_dropout=classifier_dropout,
        )

        # saving all hyperparameters for easy access later
        self.message_passing_steps = message_passing_steps
        self.encoder_dropout = encoder_dropout
        self.encoder_hidden_size = encoder_hidden_size
        self.add_bias_to_encoder_lazers = add_bias_to_encoder_layers
        self.aggregation = aggregation
        self.classifier_num_hidden_layers = classifier_num_hidden_layers
        self.classifier_hidden_dim = classifier_hidden_dim
        self.classifier_dropout = classifier_dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # use BCE loss with logits
        self.loss = nn.BCEWithLogitsLoss(pos_weight=Config.get_pos_weights())

        self.save_hyperparameters()

    def forward(self, molecules: List[List[Mol]]) -> torch.Tensor:
        logits = self.model(molecules)
        return logits

    def training_step(
        self, batch: Tuple[List[List[Mol]], torch.Tensor], *args: Any, **kwargs: Any
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
        self, batch: Tuple[List[List[Mol]], torch.Tensor], *args: Any, **kwargs: Any
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
        self, batch: Tuple[List[List[Mol]], torch.Tensor], *args: Any, **kwargs: Any
    ) -> None:
        x, y = batch
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
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer
