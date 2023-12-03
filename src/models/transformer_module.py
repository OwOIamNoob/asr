from collections import OrderedDict
from typing import Any, Dict

from omegaconf import DictConfig 

import torch
from torch import Tensor
import pytorch_lightning as pl

from src.models.transformer.model.encoder import Encoder
from src.models.transformer.model.decoder import Decoder

from src.models.transformer.modules.utils import get_class_name

from src.models.transformer.tokenizers.tokenizer import Tokenizer
# TODO: USE OWN TOKENIZER INSTEAD OF THE DEFAULT ONE.


class TransformerLitModule(pl.LightningModule):
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
        super(TransformerLitModule, self).__init__(configs, tokenizer)
        self.configs = configs
        self.teacher_forcing_ratio = configs.model.teacher_forcing_ratio

        self.pad_id=self.tokenizer.pad_id,
        self.sos_id=self.tokenizer.sos_id,
        self.eos_id=self.tokenizer.eos_id,

        self.encoder = Encoder(
            vocab_size=self.configs.vocab_size,
            input_dim=self.configs.audio.num_mels,
            d_model=self.configs.model.d_model,
            d_ff=self.configs.model.d_ff,
            num_layers=self.configs.model.num_encoder_layers,
            num_heads=self.configs.model.num_attention_heads,
            dropout_p=self.configs.model.encoder_dropout_p,
        )

        self.decoder = Decoder(
            vocab_size=self.configs.vocab_size,
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

    def set_beam_decoder(self, beam_size: int = 3):
        """
        Set the decoder to do beam search
        
        NOTE: THIS IS A ONE-TIME OPERATION
        """
        from src.models.transformer.model.beam_search_decoder import BeamSearchDecoder
        
        self.decoder = BeamSearchDecoder(self.decoder, beam_size=beam_size)
    
    def collect_outputs(
        self,
        stage: str,
        logits: Tensor,
        encoder_logits: Tensor,
        encoder_output_lengths: Tensor,
        targets: Tensor,
        target_lengths: Tensor,
    ) -> OrderedDict:
        loss = self.criterion(logits, targets[:, 1:])
        self.info({f"{stage}_loss": loss})

        predictions = logits.max(-1)[1]

        return OrderedDict(
            {
                "loss": loss,
                "predictions": predictions,
                "targets": targets,
                "logits": logits,
            }
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Dict[str, Tensor]: 
        r"""
        Forward propagate a `inputs` and `targets` pair for inference.

        Inputs:
            inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            outputs (dict): Result of model predictions that contains `predictions`, `logits`, `encoder_outputs`,
                `encoder_logits`, `encoder_output_lengths`.
        """
        logits = None
        encoder_outputs, encoder_logits, encoder_output_lengths = self.encoder(inputs, input_lengths)

        if get_class_name(self.decoder) == "BeamSearchDecoder":
            predictions = self.decoder(encoder_outputs, encoder_output_lengths)
        else:
            logits = self.decoder(
                encoder_outputs=encoder_outputs,
                encoder_output_lengths=encoder_output_lengths,
                teacher_forcing_ratio=0.0,
            )
            predictions = logits.max(-1)[1]
        return {
            "predictions": predictions,
            "logits": logits,
            "encoder_outputs": encoder_outputs,
            "encoder_logits": encoder_logits,
            "encoder_output_lengths": encoder_output_lengths,
        }
    
    def training_step(self, batch: tuple, batch_idx: int) -> OrderedDict:
        r"""
        Forward propagate a `inputs` and `targets` pair for training.

        Inputs:
            train_batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.Tensor): loss for training
        """
        inputs, targets, input_lengths, target_lengths = batch

        encoder_outputs, encoder_logits, encoder_output_lengths = self.encoder(inputs, input_lengths)
        if get_class_name(self.decoder) == "Decoder":
            logits = self.decoder(
                encoder_outputs=encoder_outputs,
                targets=targets,
                encoder_output_lengths=encoder_output_lengths,
                target_lengths=target_lengths,
                teacher_forcing_ratio=self.teacher_forcing_ratio,
            )
        else:
            raise ValueError("Why is your decoder not a Decoder?")

        return self.collect_outputs(
            stage="train",
            logits=logits,
            encoder_logits=encoder_logits,
            encoder_output_lengths=encoder_output_lengths,
            targets=targets,
            target_lengths=target_lengths,
        )
    
    def validation_step(self, batch: tuple, batch_idx: int) -> OrderedDict:
        r"""
        Forward propagate a `inputs` and `targets` pair for validation.

        Inputs:
            train_batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.Tensor): loss for training
        """
        inputs, targets, input_lengths, target_lengths = batch

        encoder_outputs, encoder_logits, encoder_output_lengths = self.encoder(inputs, input_lengths)
        logits = self.decoder(
            encoder_outputs,
            encoder_output_lengths=encoder_output_lengths,
            teacher_forcing_ratio=0.0,
        )
        return self.collect_outputs(
            stage="val",
            logits=logits,
            encoder_logits=encoder_logits,
            encoder_output_lengths=encoder_output_lengths,
            targets=targets,
            target_lengths=target_lengths,
        )
    
    def test_step(self, batch: tuple, batch_idx: int) -> OrderedDict:
        r"""
        Forward propagate a `inputs` and `targets` pair for test.

        Inputs:
            train_batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.Tensor): loss for training
        """
        inputs, targets, input_lengths, target_lengths = batch

        encoder_outputs, encoder_logits, encoder_output_lengths = self.encoder(inputs, input_lengths)
        logits = self.decoder(
            encoder_outputs,
            encoder_output_lengths=encoder_output_lengths,
            teacher_forcing_ratio=0.0,
        )
        return self.collect_outputs(
            stage="test",
            logits=logits,
            encoder_logits=encoder_logits,
            encoder_output_lengths=encoder_output_lengths,
            targets=targets,
            target_lengths=target_lengths,
        )

      