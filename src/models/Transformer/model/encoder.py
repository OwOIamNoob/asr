
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from Transformer.layers.encoder_layer import EncoderLayer
from Transformer.modules.wrapper import Linear
from Transformer.modules.positional_encoding import PositionalEncoding


class Encoder(nn.Module):
    r"""
    The TransformerEncoder is composed of a stack of N identical layers.
    Each layer has two sub-layers. The first is a multi-head self-attention mechanism,
    and the second is a simple, position-wise fully connected feed-forward network.

    Args:
        input_dim: dimension of feature vector
        d_model: dimension of model (default: 512)
        d_ff: dimension of feed forward network (default: 2048)
        num_layers: number of encoders layers (default: 6)
        num_heads: number of attention heads (default: 8)
        dropout_p:  probability of dropout (default: 0.3)
        max_len: maximum sequence length (default: 5000)

    Inputs:
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is list of tokens
        - **src_mask**: mask of source language

    Returns:
        * outputs: A output sequence of encoders. `FloatTensor` of size ``(batch, seq_length, dimension)``
    """

    def __init__(
        self,
        input_dim: int = 80,
        d_model: int = 512,
        d_ff: int = 2048,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout_p: float = 0.3,
        max_length: int = 5000
    ) -> None:
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.input_proj = Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        self.positional_encoding = PositionalEncoding(d_model, max_length=max_length)

        self.input_dropout = nn.Dropout(p=dropout_p)

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
        inputs: Tensor,
        src_mask: Tensor
    ) -> Tensor:
        r"""
        Forward propagate `inputs` for  encoders training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            src_mask (torch.BoolTensor): mask of source language

        Returns:
            * outputs: A output sequence of encoders. `FloatTensor` of size ``(batch, seq_length, dimension)``
        """

        outputs = self.input_norm(self.input_proj(inputs))
        outputs += self.positional_encoding(outputs.size(1))
        outputs = self.input_dropout(outputs)

        for layer in self.layers:
            outputs = layer(outputs, src_mask)

        return outputs
