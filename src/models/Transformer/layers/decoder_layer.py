from typing import Optional, Tuple

import torch.nn as nn
from torch import Tensor

from Transformer.modules.multi_head_attention import MultiHeadAttention
from Transformer.modules.positionwise_feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    r"""
    DecoderLayer is made up of self-attention, multi-head attention and feedforward network.
    This standard decoders layer is based on the paper "Attention Is All You Need".

    Args:
        d_model: dimension of model (default: 512)
        num_heads: number of attention heads (default: 8)
        d_ff: dimension of feed forward network (default: 2048)
        dropout_p: probability of dropout (default: 0.3)

    Inputs:
        inputs (torch.FloatTensor): input sequence of transformer decoder layer
        encoder_outputs (torch.FloatTensor): outputs of encoder
        self_attn_mask (torch.BoolTensor): mask of self attention
        encoder_output_mask (torch.BoolTensor): mask of encoder outputs

    Returns:
        * outputs (torch.FloatTensor): output of transformer decoder layer
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout_p: float = 0.3,
    ) -> None:
        super(DecoderLayer, self).__init__()
        self.self_attention_prenorm = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.self_attention_postnorm = nn.LayerNorm(d_model)

        self.decoder_attention_prenorm = nn.LayerNorm(d_model)
        self.decoder_attention = MultiHeadAttention(d_model, num_heads)
        self.decoder_attention_postnorm = nn.LayerNorm(d_model)

        self.feed_forward_prenorm = nn.LayerNorm(d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_p)
        self.feed_forward_postnorm = nn.LayerNorm(d_model)

    def forward(
        self,
        inputs: Tensor,
        encoder_outputs: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor
    ) -> Tensor:
        r"""
        Forward propagate transformer decoder layer.

        Inputs:
            inputs (torch.FloatTensor): input sequence of transformer decoder layer
            encoder_outputs (torch.FloatTensor): outputs of encoder
            src_mask (torch.BoolTensor): mask of source language
            tgt_mask (torch.BoolTensor): mask of target language

        Returns:
            outputs (torch.FloatTensor): output of transformer decoder layer
        """
        residual = inputs
        inputs = self.self_attention_prenorm(inputs)
        outputs = self.self_attention(inputs, inputs, inputs, tgt_mask)
        outputs = self.self_attention_postnorm(outputs)
        outputs += residual

        residual = outputs
        outputs = self.decoder_attention_prenorm(outputs)
        outputs = self.decoder_attention(outputs, encoder_outputs, encoder_outputs, src_mask)
        outputs = self.decoder_attention_postnorm(outputs)
        outputs += residual

        residual = outputs
        outputs = self.feed_forward_prenorm(outputs)
        outputs = self.feed_forward(outputs)
        outputs = self.feed_forward_postnorm(outputs)
        outputs += residual

        return outputs