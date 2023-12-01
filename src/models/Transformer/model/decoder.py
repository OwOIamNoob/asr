# MIT License
#
# Copyright (c) 2021 Soohwan Kim and Sangchun Ha and Soyoung Cho
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import random
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


from Transformer.layers.decoder_layer import DecoderLayer
from Transformer.modules.wrapper import Linear
from Transformer.modules.positional_encoding import PositionalEncoding
from Transformer.modules.transformer_embedding import TransformerEmbedding


class Decoder(nn.Module):
    r"""
    The TransformerDecoder is composed of a stack of N identical layers.
    Each layer has three sub-layers. The first is a multi-head self-attention mechanism,
    and the second is a multi-head attention mechanism, third is a feed-forward network.

    Args:
        vocab_size: vocabulary_size
        d_model: dimension of model
        d_ff: dimension of feed forward network
        num_layers: number of layers
        num_heads: number of attention heads
        dropout_p: probability of dropout
        pad_id: index of the pad symbol (default: 0)
        sos_id: index of the start of sentence symbol (default: 1)
        eos_id: index of the end of sentence symbol (default: 2)        
        max_len: maximum sequence length (default: 5000)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        d_ff: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout_p: float = 0.3,
        pad_id: int = 0,
        sos_id: int = 1,
        eos_id: int = 2,
        max_length: int = 5000,
    ) -> None:
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_length = max_length

        self.embedding = TransformerEmbedding(vocab_size, pad_id, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.input_dropout = nn.Dropout(p=dropout_p)
        
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout_p=dropout_p,
                )
                for _ in range(num_layers)
            ]
        )

        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            Linear(d_model, vocab_size, bias=False),
        )

    def forward_step(
        self,
        decoder_inputs: Tensor,
        encoder_outputs: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
        positional_encoding_length: int
    ) -> Tensor:
        outputs = self.embedding(decoder_inputs) + self.positional_encoding(positional_encoding_length)
        outputs = self.input_dropout(outputs)

        for layer in self.layers:
            outputs = layer(
                inputs=outputs,
                encoder_outputs=encoder_outputs,
                src_mask=src_mask,
                tgt_mask=tgt_mask
            )

        outputs = self.fc(outputs)

        return outputs

    def forward(
        self,
        targets: torch.LongTensor,
        encoder_outputs: torch.Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
        teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        r"""
        Forward propagate a `encoder_outputs` for training.

        Args:
            targets (torch.LongTensor): A target sequence passed to decoders. `IntTensor` of size
                ``(batch, seq_length)``
            encoder_outputs (torch.FloatTensor): A output sequence of encoders. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            src_mask (torch.BoolTensor): mask of source language
            tgt_mask (torch.BoolTensor): mask of target language
            teacher_forcing_ratio (float): ratio of teacher forcing

        Returns:
            * logits (torch.FloatTensor): Log probability of model predictions.
        """
        logits = list()
        batch_size = encoder_outputs.size(0)
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            pass
        # Inference
        else:
            pass
        
        raise NotImplementedError
