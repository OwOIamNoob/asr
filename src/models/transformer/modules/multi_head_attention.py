from typing import Optional, Tuple

import torch.nn as nn
from torch import Tensor

from src.models.transformer.modules.dot_product_attention import DotProductAttention
from src.models.transformer.modules.wrapper import Linear


class MultiHeadAttention(nn.Module):
    r"""
    Multi-Head Attention in "Attention Is All You Need" (section 3.2.2).

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_o
        where head_i = Attention(Q · W_q, K · W_k, V · W_v)

    Args:
        d_model (int): The dimension of model 
        num_heads (int): The number of attention heads. 

    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoders.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoders.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (batch, _len, _len): tensor containing indices to be masked

    Returns: output
        - **output** (batch, output_len, dimensions): tensor containing the attended output features.
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "hidden_dim % num_heads should be zero."

        self.d_head = d_model // num_heads
        self.num_heads = num_heads

        self.query_proj = Linear(d_model, self.d_head * num_heads)
        self.key_proj = Linear(d_model, self.d_head * num_heads)
        self.value_proj = Linear(d_model, self.d_head * num_heads)

        self.scaled_dot_attn = DotProductAttention(d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        batch_size = value.size(0)

        # (batch, _len, d_model) -> (batch, d_model, num_heads, d_head) -> (batch, num_heads, d_model, d_head)
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)

        if mask is not None:
            # (batch, _len, _len) -> (batch, num_heads, _len, _len)
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        context, attn = self.scaled_dot_attn(query, key, value, mask)

        # (batch, num_heads, _len, d_head) -> (batch, _len, num_heads, d_head) -> (batch, _len, d_model)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.d_head)

        return context, attn