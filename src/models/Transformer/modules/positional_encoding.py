import math

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    r"""
    Positional Encoding in "Attention Is All You Need" (section 3.5).

    "Attention Is All You Need" uses sine and cosine functions of different frequencies:
        PE_(pos, 2i)    =  sin(pos / power(10000, 2i / d_model))
        PE_(pos, 2i+1)  =  cos(pos / power(10000, 2i / d_model))
    only change is that calculations are done with -log(power(10000, 2i / d_model))
    """

    def __init__(self, d_model: int, max_length: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_length, d_model, requires_grad=False)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, length: int) -> Tensor:
        return self.pe[:, :length]