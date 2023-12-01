from collections import OrderedDict
from typing import Any

from omegaconf import DictConfig # This isn't even in the orginal folder

import torch
import pytorch_lightning as pl

from Transformer.model.encoder import Encoder
from Transformer.model.decoder import Decoder
from Transformer.tokenizers.tokenizer import Tokenizer


class TransformerModel(pl.LightningModule):
    r"""
    A Transformer model. User is able to modify the attributes as needed.
    The model is based on the paper "Attention Is All You Need".

    Args:
        configs (DictConfig): configuration set.
        tokenizer (Tokeizer): tokenizer is in charge of preparing the inputs for a model.

    Inputs:
        - **inputs** (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
        - **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``

    Returns:
        outputs (dict): Result of model predictions.
    """

    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(TransformerModel, self).__init__(configs, tokenizer)

        self.pad_id=self.tokenizer.pad_id,
        self.sos_id=self.tokenizer.sos_id,
        self.eos_id=self.tokenizer.eos_id,

        self.encoder = Encoder(
            input_dim=self.configs.audio.num_mels,
            d_model=self.configs.model.d_model,
            d_ff=self.configs.model.d_ff,
            num_layers=self.configs.model.num_encoder_layers,
            num_heads=self.configs.model.num_attention_heads,
            dropout_p=self.configs.model.encoder_dropout_p,
            max_length=self.configs.model.max_length
        )
        self.decoder = Decoder(
            vocab_size=self.vocab_size,
            d_model=self.configs.model.d_model,
            d_ff=self.configs.model.d_ff,
            num_layers=self.configs.model.num_decoder_layers,
            num_heads=self.configs.model.num_attention_heads,
            dropout_p=self.configs.model.decoder_dropout_p,
            pad_id=self.tokenizer.pad_id,
            sos_id=self.tokenizer.sos_id,
            eos_id=self.tokenizer.eos_id,
            max_length=self.configs.model.max_length,
        )

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output
    
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def set_beam_decoder(self, beam_size: int = 3):
        """Setting beam search decoder"""
        # from Transformer.models import BeamSearchTransformer

        # self.decoder = BeamSearchTransformer(
        #     decoder=self.decoder,
        #     beam_size=beam_size,
        # )
        raise NotImplementedError