from typing import Any, Tuple, Dict, Optional
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import lightning as L


class TransformerEncoder(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 maximum_sequence_length: int,
                 embedding_dimension: int = 64,
                 number_of_heads: int = 8,
                 dimension_of_feedforward: int = 2048,
                 dropout: float = 0.1,
                 norm_first: bool = True,
                 number_of_encoder_layers: int = 3,
                 *args, **kwargs) -> None:
        
        super().__init__(*args, **kwargs)
        # nn.Embedding is like a one-hot encoding layer
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dimension)
        # here we use one hot encoding for the position embedding
        self.position_embedding_table = nn.Embedding(num_embeddings=maximum_sequence_length, embedding_dim=embedding_dimension)
        # defining encoding
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dimension, nhead=number_of_heads, dim_feedforward=dimension_of_feedforward, dropout=dropout, batch_first=True, norm_first=norm_first)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=number_of_encoder_layers)
        # normalizes over the last dimension with trainable parameters
        self.layernorm = nn.LayerNorm(embedding_dimension)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x is of shape (batch_size, sequence_length)
        B, S = x.shape
        # embed the tokens TODO: Should I scale here?
        token_embedding = self.token_embedding_table(x) # now of shape batch_size, sequence_length, embedding_dimension (B, S, E)
        position_embedding = self.position_embedding_table(torch.arange(S, device=x.device)) # has shape (S, E)
        # forward pass
        out = token_embedding + position_embedding # (B, S, E)
        # TODO: Should I apply dropout here?
        out = self.encoder(out, src_key_padding_mask= padding_mask) # (B, S, E)
        # normalize x across the embedding dimension per token
        out = self.layernorm(out) # (B, S, E)        
        return out # (B, S, E)
        
class TransformerDecoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 maximum_sequence_length: int,
                 embedding_dimension: int = 64,
                 number_of_heads: int = 8,
                 dimension_of_feedforward: int = 2048,
                 dropout: float = 0.1,
                 norm_first: bool = True,
                 number_of_decoder_layers: int = 3,
                 *args, **kwargs) -> None:
        
        super().__init__(*args, **kwargs)
        # create embedding tables for target tokens and positions
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dimension)
        self.position_embedding_table = nn.Embedding(num_embeddings=maximum_sequence_length, embedding_dim=embedding_dimension)
        # defining the decoding
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dimension, nhead=number_of_heads, dim_feedforward=dimension_of_feedforward, dropout=dropout, batch_first=True, norm_first=norm_first)
        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=number_of_decoder_layers)
        self.decoderOutLayer = nn.Linear(embedding_dimension, vocab_size)
        
        
    def forward(self, encoder_output: torch.Tensor, target: torch.Tensor, target_mask: Optional[torch.Tensor] = None, target_padding_mask: Optional[torch.Tensor] = None, memory_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # embedding the target (batch_size x target_sequence_length) (B, T)
        T = target.shape[1]
        # TODO: Layernorm here?
        token_embedding = self.token_embedding_table(target) # (B, T, E)
        position_embedding = self.position_embedding_table(torch.arange(T, device=target.device)) # (T, E)
        # forward pass
        out = token_embedding + position_embedding # (B, T, E)
        # TODO: Dropout here?
        out = self.decoder(tgt=out, memory=encoder_output, tgt_mask=target_mask, tgt_key_padding_mask = target_padding_mask, memory_key_padding_mask = memory_padding_mask) # (B, T, E)
        out = self.decoderOutLayer(out) # (B, T, vocab_size)
        return out # (B, T, vocab_size)
    

class SeqToSeqTransformerMatrix(L.LightningModule):
    def __init__(self,
                 char2idx: Dict,
                 maximum_input_sequence_length: int = 202, # 200 + SOS + EOS
                 maximum_target_sequence_length: int = 203, # 200 + SOS + EOS + PAD (to allow frameshift)
                 embedding_dimension: int = 64,
                 number_of_heads: int = 8,
                 dimension_of_feedforward: int = 2048,
                 dropout: float = 0.1,
                 norm_first: bool = True,
                 number_of_encoder_layers: int = 3,
                 number_of_decoder_layers: int = 3,
                 learning_rate: float = 1e-3,
                 *args: Any, **kwargs: Any) -> None:
        
        super().__init__(*args, **kwargs)
        
        # safe char2idx dict for later uses
        self.char2idx: Dict[str, int] = char2idx
        self.idx2char: Dict[int, str] = {idx: char for char, idx in char2idx.items()}
        # get vocab_size
        vocab_size = len(char2idx)
        self.vocab_size = vocab_size
        # save embedding dimension
        self.embedding_dimension = embedding_dimension
        # Encoder
        self.encoder = TransformerEncoder(vocab_size=vocab_size, maximum_sequence_length=maximum_input_sequence_length, embedding_dimension=embedding_dimension, number_of_heads=number_of_heads, dimension_of_feedforward=dimension_of_feedforward, dropout=dropout, norm_first=norm_first, number_of_encoder_layers=number_of_encoder_layers)
        # Decoder 
        self.decoder = TransformerDecoder(vocab_size=vocab_size, maximum_sequence_length=maximum_target_sequence_length, embedding_dimension=embedding_dimension, number_of_heads=number_of_heads, dimension_of_feedforward=dimension_of_feedforward, dropout=dropout, norm_first=norm_first, number_of_decoder_layers=number_of_decoder_layers)
        # loss (ignores padding index 0 and the loss is averaged over the non-padded tokens to make it comparable across batches of different sizes)
        self.loss = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
        # # learning rate
        self.learning_rate = learning_rate

        # logging hyperparameters
        self.save_hyperparameters()
        
    def forward(self, x: torch.Tensor, target: torch.Tensor, input_padding_mask: Optional[torch.Tensor] = None, target_mask: Optional[torch.Tensor] = None, target_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x is of shape (batch_size, sequence_length)
        # encoding
        encoder_output = self.encoder(x=x, padding_mask=input_padding_mask) # (B, S, E)
        # decoding
        out = self.decoder(encoder_output=encoder_output, target=target, target_mask=target_mask, target_padding_mask=target_padding_mask, memory_padding_mask=input_padding_mask) # (B, T, vocab_size)
        return out, encoder_output 
    
    def create_padding_mask(self, x: torch.Tensor, padding_value: int = 0) -> torch.Tensor:
        # x is of shape (batch_size, sequence_length)
        # Note: [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked positions. If a BoolTensor is provided, positions with True are not allowed to attend while False values will be unchanged.
        mask = (x == padding_value) # (B, S)
        return mask
    
    def create_target_mask(self, target: torch.Tensor) -> torch.Tensor:
        # target is of shape (batch_size, target_length)
        # creates mask with -inf in upper triangle and 0 in lower triangle
        T = target.shape[1]
        mask = ~(torch.tril(torch.ones(T, T, device=target.device)) == 1) # (T, T)
        return mask
        
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        def lr_lambda(current_step: int) -> float:
            lambda_val = 20 * min(1.0, (float(current_step)/ 16000)) / max(current_step, 16000)
            return max(lambda_val/self.learning_rate, 1e-4/self.learning_rate) # divide by learning rate because scheduler multiplies with learning rate

        scheduler = {
            'scheduler': LambdaLR(optimizer, lr_lambda),
            'interval': 'step',
            'frequency': 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def training_step(self, batch, batch_idx):
        input, target = batch
        
        # shift target by one token to the right and prepend start of sequence token
        target_shifted = torch.cat([torch.ones(target.shape[0], 1, dtype=torch.long, device=target.device) * self.char2idx['<SOS>'], target[:, :-1]], dim=1) # (B, T)
        # generate masks
        input_padding_mask = self.create_padding_mask(input)
        target_padding_mask = self.create_padding_mask(target_shifted)
        target_mask = self.create_target_mask(target_shifted)
        # forward pass
        output, _ = self.forward(x=input, input_padding_mask=input_padding_mask, target=target_shifted, target_mask=target_mask, target_padding_mask=target_padding_mask)
        # reshape output and target for the loss function
        output = output.view(-1, output.shape[-1]) # (B*T, vocab_size)
        target = target.view(-1) # (B*T)
        # calculate loss
        loss = self.loss(output, target)
        # log loss
        self.log('average_train_loss_per_token', loss)
        # return loss
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        input, target = batch

        # shift target by one token to the right and prepend start of sequence token
        target_shifted = torch.cat([torch.ones(target.shape[0], 1, dtype=torch.long, device=target.device) * self.char2idx['<SOS>'], target[:, :-1]], dim=1) # (B, T)        
        # generate masks
        input_padding_mask = self.create_padding_mask(input)
        target_padding_mask = self.create_padding_mask(target_shifted)
        target_mask = self.create_target_mask(target_shifted)
        # forward pass
        output, _ = self.forward(x=input, input_padding_mask=input_padding_mask, target=target_shifted, target_mask=target_mask, target_padding_mask=target_padding_mask)
        # reshape output and target for the loss function
        output = output.view(-1, output.shape[-1]) # (B*T, vocab_size)
        target = target.view(-1) # (B*T)
        # calculate loss
        loss = self.loss(output, target)
        # log loss
        self.log('average_validation_loss_per_token', loss)
        # Calculate accuracy per token
        output = output.argmax(dim=-1) # (B*T)
        non_pad_mask = target != self.char2idx['<PAD>']
        num_correct = (output == target).masked_select(non_pad_mask).sum().item()
        num_total = non_pad_mask.sum().item()
        accuracy = num_correct / num_total if num_total != 0 else 0
        # log accuracy
        self.log('average_validation_accuracy_per_token', accuracy)
    
    def test_step(self, batch, batch_idx):
        input, target = batch

        # shift target by one token to the right and prepend start of sequence token
        target_shifted = torch.cat([torch.ones(target.shape[0], 1, dtype=torch.long, device=target.device) * self.char2idx['<SOS>'], target[:, :-1]], dim=1) # (B, T)        
        # generate masks
        input_padding_mask = self.create_padding_mask(input)
        target_padding_mask = self.create_padding_mask(target_shifted)
        target_mask = self.create_target_mask(target_shifted)
        # forward pass
        output, _ = self.forward(x=input, input_padding_mask=input_padding_mask, target=target_shifted, target_mask=target_mask, target_padding_mask=target_padding_mask)
        # reshape output and target for the loss function
        output = output.view(-1, output.shape[-1]) # (B*T, vocab_size)
        target = target.view(-1) # (B*T)
        # calculate loss
        loss = self.loss(output, target)
        # log loss
        self.log('average_test_loss_per_token', loss)

    
    @torch.no_grad()
    def get_encoding_from_sequence(self, sequence: str) -> torch.Tensor:
        """
        Input: raw smiles sequence
        Output: Latent encoding of the transformer encoder
        """
        assert isinstance(sequence, str), f'Sequence must be of type str, not {type(sequence)}'
        integers = self.get_integers_from_sequence(sequence)
        encoding = self.get_encoding_from_integers(integers)
        return encoding
    
    @torch.no_grad()
    def get_integers_from_sequence(self, sequence:str) -> torch.Tensor:
        """
        Input: raw smiles sequence
        Output: tensor of integers from char2idx
        """
        assert isinstance(sequence, str), f'Sequence must be of type str, not {type(sequence)}'
        integers = torch.tensor([self.char2idx['<SOS>']]+[self.char2idx.get(char, self.char2idx['<UNK>']) for char in sequence]+[self.char2idx['<EOS>']], dtype=torch.long, device=self.device)
        return integers
    
    @torch.no_grad()
    def get_encoding_from_integers(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Input: tensor of integers from char2idx
        Output: Latent encoding of the transformer encoder
        """
        if len(sequence.shape) == 1:
            sequence = sequence.unsqueeze(0)
        if len(sequence.shape) != 2:
            raise ValueError(f'Sequence must be of shape (batch_size, sequence_length) or (sequence_length,) not of shape {sequence.shape}')
        input_padding_mask = self.create_padding_mask(sequence)
        # encoding
        encoder_output = self.encoder(x=sequence, padding_mask=input_padding_mask)
        return encoder_output # (B, S, E)
    

        