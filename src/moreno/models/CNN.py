# CNN to work on top of the transformer
from typing import List
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

class HighwayUnit(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.T = nn.Linear(in_features, out_features)
        self.H = nn.Linear(in_features, out_features)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        T = torch.sigmoid(self.T(input))
        H = F.relu(self.H(input))
        return T * H + (1 - T) * input  # (batch_size, out_features)

class Model(nn.Module):

    def __init__(self,
                 filter_sizes: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
                 n_filters: List[int] = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160],
                 embedding_dim: int = 64,
                 dropout: float = 0.25, *args, **kwargs
                 ) -> None:
        # Create the convolutional layers
        super().__init__(*args, **kwargs)
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1, 
                out_channels=n_filters[i],
                kernel_size=(filter_sizes[i], embedding_dim)
            ) for i in range(len(filter_sizes))])
        self.dropoutLayer = nn.Dropout(dropout)
        self.fc1 = nn.Linear(sum(n_filters), 512)
        self.highway = HighwayUnit(512, 512)
        self.fc2 = nn.Linear(512, 1)        
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
    # input shape: (batch_size, seq_len, embedding_dim)
        input = input.unsqueeze(1) # (batch_size, 1, seq_len, embedding_dim)
        conved = [F.relu(conv(input)).squeeze(3) for conv in self.convs] # [(batch_size, n_filters, seq_len - filter_sizes[n] + 1), ...]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved] # [(batch_size, n_filters), ...]
        out = self.dropoutLayer(torch.cat(pooled, dim=1)) # (batch_size, sum(n_filters[i])
        out = F.relu(self.fc1(out)) # (batch_size, 512)
        out = self.highway(out) # (batch_size, 512)
        out = self.fc2(out) # (batch_size, 1)
        return out      

class CNNLightning(L.LightningModule):
    def __init__(self,
                 filter_sizes: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
                 n_filters: List[int] = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160],
                 embedding_dim: int = 64,
                 dropout: float = 0.25,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 ) -> None:
        super().__init__()

        # ensuring correct input
        if len(filter_sizes) != len(n_filters):
            raise ValueError("filter_sizes and n_filters must have the same length")
        
        # saving the configuration
        self.filter_sizes = filter_sizes
        self.n_filters = n_filters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Create the convolutional layers
        self.model = Model(filter_sizes=filter_sizes,
                           n_filters=n_filters,
                           embedding_dim=embedding_dim,
                           dropout=dropout,
                           )
        self.loss = nn.BCEWithLogitsLoss()

        # save hyperparameters
        self.save_hyperparameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input shape: (batch_size, seq_len, embedding_dim)
        out = self.model(input)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch # (batch_size, seq_len, embedding_dim), (batch_size)
        if x.shape[1] < max(self.filter_sizes):
            x = F.pad(x, (0, 0, 0, max(self.filter_sizes) - x.shape[1]))
            y = F.pad(y, (0, max(self.filter_sizes) - x.shape[1]))
        y_hat = self(x).squeeze(1) # (batch_size)
        loss = self.loss(y_hat, y.float())
        # log loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch # (batch_size, seq_len, embedding_dim), (batch_size)
        if x.shape[1] < max(self.filter_sizes):
            x = F.pad(x, (0, 0, 0, max(self.filter_sizes) - x.shape[1]))
            y = F.pad(y, (0, max(self.filter_sizes) - x.shape[1]))
        y_hat = self(x).squeeze(1) # (batch_size)
        loss = self.loss(y_hat, y.float())
        # log loss
        self.log('val_loss', loss)
        # log accuracy and uncertainty
        y_hat_rounded = torch.round(torch.sigmoid(y_hat))
        accuracy = torch.sum(y_hat_rounded == y) / len(y)
        self.log('validation_accuracy', accuracy)
        return y_hat, y # (batch_size), (batch_size), 

    def test_step(self, batch, batch_idx):
        x, y = batch # (batch_size, seq_len, embedding_dim), (batch_size)
        if x.shape[1] < max(self.filter_sizes):
            x = F.pad(x, (0, 0, 0, max(self.filter_sizes) - x.shape[1]))
            y = F.pad(y, (0, max(self.filter_sizes) - x.shape[1]))
        y_hat = self(x).squeeze(1) # (batch_size)
        loss = self.loss(y_hat, y.float())
        # log loss
        self.log('test_loss', loss)
        # log accuracy and uncertainty
        y_hat_rounded = torch.round(torch.sigmoid(y_hat))
        accuracy = torch.sum(y_hat_rounded == y) / len(y)
        self.log('test_accuracy', accuracy)
        return y_hat, y # (batch_size), (batch_size), for AUC ROC curve(TODO)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer
