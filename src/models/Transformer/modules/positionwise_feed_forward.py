
import torch.nn as nn
from torch import Tensor
from Transformer.modules.wrapper import Linear


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feedforward Networks in "Attention Is All You Need" (section 3.3).
    Basically a network with 2 fully-connected layers with ReLU activation in between.
    """

    def __init__(self, d_model: int, d_ff: int, dropout_p: float) -> None:
        super(PositionwiseFeedForward, self).__init__()
        self.feed_forward = nn.Sequential(
            Linear(d_model, d_ff),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            Linear(d_ff, d_model),
            nn.Dropout(dropout_p),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.feed_forward(x)