
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DotProductAttention(nn.Module):
    """
    (Scaled) Dot-Product Attention in "Attention Is All You Need" (section 3.2.1).
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values

    Args: dim (int): dimension of attention - usually equals to d_model

    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoders.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoders.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked

    Returns: 
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoders outputs.
    """

    def __init__(self, dim: int) -> None:
        super(DotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if len(query.size()) == 3:# (num_heads, d_model, d_head).
            score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        else: # (batch, num_heads, d_model, d_head).
            score = torch.matmul(query, key.transpose(2, 3)) / self.sqrt_dim

        if mask is not None:
            score.masked_fill_(mask, -1e4) # -1e9 if it isn't small enough.

        attn = F.softmax(score, -1)

        
        if len(query.size()) == 3: # (num_heads, d_model, d_head).
            context = torch.bmm(attn, value)
        else: # (batch, num_heads, d_model, d_head).
            context = torch.matmul(attn, value)

        return context, attn
