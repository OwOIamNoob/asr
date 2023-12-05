# Wrapper classes, slightly changing certain things

import torch.nn as nn
from torch import Tensor

class Linear(nn.Module):
    r"""
    Wrapper class of torch.nn.Linear to initialize weights by xavier initialization and biases to zeros.
    """

    def __init__(self, in_features: int, out_features: int, zero_bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.linear.weight)
        if zero_bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)
