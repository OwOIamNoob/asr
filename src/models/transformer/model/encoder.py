
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.models.transformer.layers.encoder_layer import EncoderLayer
from src.models.transformer.modules.wrapper import Linear
from src.models.transformer.modules.positional_encoding import PositionalEncoding
from src.models.transformer.modules.mask import get_attn_pad_mask


class Encoder(nn.Module):
    r"""
    The TransformerEncoder is composed of a stack of N identical layers.

    Args:
        input_dim: dimension of feature vector
        d_model: dimension of model (default: 512)
        d_ff: dimension of feed forward network (default: 2048)
        num_layers: number of encoders layers (default: 6)
        num_heads: number of attention heads (default: 8)
        dropout_p:  probability of dropout (default: 0.3)

    Inputs:
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is list of tokens
        - **input_lengths**: list of sequence lengths

    Returns:
        (Tensor, Tensor):

        * outputs: A output sequence of encoders. `FloatTensor` of size ``(batch, seq_length, dimension)``
        * output_lengths: The length of encoders outputs. ``(batch)``
    """

    def __init__(
        self,
        vocab_size: int,
        input_dim: int = 80,
        d_model: int = 512,
        d_ff: int = 2048,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout_p: float = 0.3,
        pad_id: int = 2
    ) -> None:
        super(Encoder, self).__init__()

        self.vocab_size = vocab_size

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        self.embedding = nn.Embedding(vocab_size, pad_id, input_dim)
        self.input_proj = Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(p=dropout_p)
        
        self.positional_encoding = PositionalEncoding(d_model)

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout_p=dropout_p,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Forward propagate a `inputs` for  encoders training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            (Tensor, Tensor):

            * outputs: A output sequence of encoders. `FloatTensor` of size ``(batch, seq_length, dimension)``
            * output_lengths: The length of encoders outputs. ``(batch)``
        """
        self_attn_mask = get_attn_pad_mask(inputs, input_lengths, inputs.size(1))

        outputs = self.input_norm(self.input_proj(self.embedding(inputs)))
        outputs += self.positional_encoding(outputs.size(1))
        outputs = self.input_dropout(outputs)

        for layer in self.layers:
            outputs, attn = layer(outputs, self_attn_mask)

        return outputs, input_lengths

    
    def count_parameters(self) -> int:
        r"""Count parameters of encoders"""
        return sum([p.numel for p in self.parameters()])

    def update_dropout(self, dropout_p: float) -> None:
        r"""Update dropout probability of encoders"""
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p
