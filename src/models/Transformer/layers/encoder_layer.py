
from typing import Tuple

import torch.nn as nn
from torch import Tensor

from Transformer.modules.multi_head_attention import MultiHeadAttention
from Transformer.modules.positionwise_feed_forward import PositionwiseFeedForward

class EncoderLayer(nn.Module):
    r"""
    EncoderLayer is made up of self-attention and feedforward network.

    Args:
        d_model: dimension of model (default: 512)
        num_heads: number of attention heads (default: 8)
        d_ff: dimension of feed forward network (default: 2048)
        dropout_p: probability of dropout (default: 0.3)

    Inputs:
        inputs (torch.FloatTensor): input sequence of transformer encoder layer
        src_mask (torch.BoolTensor): mask of source language

    Returns:
        * outputs (torch.FloatTensor): output of transformer encoder layer
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout_p: float = 0.3,
    ) -> None:
        super(EncoderLayer, self).__init__()
        self.attention_prenorm = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, num_heads)

        self.feed_forward_prenorm = nn.LayerNorm(d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_p)
        

    def forward(self, inputs: Tensor, src_mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        r"""
        Forward propagate of transformer encoder layer.

        Inputs:
            inputs (torch.FloatTensor): input sequence of transformer encoder layer
            self_attn_mask (torch.BoolTensor): mask of self attention

        Returns:
            outputs (torch.FloatTensor): output of transformer encoder layer
        """
        residual = inputs
        inputs = self.attention_prenorm(inputs)
        outputs, attn = self.self_attention(inputs, inputs, inputs, src_mask)
        outputs += residual

        residual = outputs
        outputs = self.feed_forward_prenorm(outputs)
        outputs = self.feed_forward(outputs)
        outputs += residual

        return outputs

