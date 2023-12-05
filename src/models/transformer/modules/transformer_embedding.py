   
import math

import torch.nn as nn
from torch import Tensor


class TransformerEmbedding(nn.Module):
    r"""
    Input Embeddings in "Attention Is All You Need" (section 3.4).
    Multiplies pytorch's in-built nn.Embedding with sqrt(d_model)

    Args:
        num_embeddings (int): the number of embedding size
        d_model (int): dimension of model

    Inputs:
        inputs (torch.FloatTensor): input of embedding layer

    Returns:
        outputs (torch.FloatTensor): output of embedding layer
    """

    def __init__(self, num_embeddings: int, pad_id: int, d_model: int) -> None:
        super(TransformerEmbedding, self).__init__()
        self.sqrt_dim = math.sqrt(d_model)
        self.embedding = nn.Embedding(num_embeddings, d_model, padding_idx=pad_id)

    def forward(self, inputs: Tensor) -> Tensor:
        r"""
        Forward propagate of embedding layer.

        Inputs:
            inputs (torch.FloatTensor): input of embedding layer

        Returns:
            outputs (torch.FloatTensor): output of embedding layer
        """
        return self.embedding(inputs) * self.sqrt_dim


