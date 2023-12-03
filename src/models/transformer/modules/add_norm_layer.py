
import torch.nn as nn


class AddNorm(nn.Module):
    """
    Add & Normalization layer proposed in "Attention Is All You Need".
    Transformers employ a residual connection around each of the two sub-layers,
    (Multi-Head Attention & Feed-Forward) followed by layer normalization.
    """

    def __init__(self, sublayer: nn.Module, d_model: int = 512) -> None:
        super(AddNorm, self).__init__()
        self.sublayer = sublayer
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, *args):
        residual = args[0]
        outputs = self.sublayer(*args)

        if isinstance(outputs, tuple):
            # outputs are from the attention layer
            return self.layer_norm(outputs[0] + residual), outputs[1]

        # outputs are from the feed-forward layer
        return self.layer_norm(outputs + residual)
