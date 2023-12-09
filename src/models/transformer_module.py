from collections import OrderedDict
from typing import Any, Dict

from omegaconf import DictConfig 

import torch
from torch import Tensor
import pytorch_lightning as pl

from src.models.transformer.model.encoder import Encoder
from src.models.transformer.model.decoder import Decoder

from src.models.transformer.modules.utils import get_class_name


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

    def __init__(
        self, 
        embedding,
        encoder: Encoder, 
        decoder: Decoder, 
        pad_id: int, 
        sos_id: int,
        eos_id: int,
        teacher_forcing_ratio: float,
    ) -> None:
        super(TransformerLitModule, self).__init__()
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id)
        
        self.pad_id=pad_id,
        self.sos_id=sos_id,
        self.eos_id=eos_id,

        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder

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
            inputs (torch.FloatTensor): A input sequence passed to encoders. 
            Typically for inputs this will be a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            outputs (dict): Result of model predictions that contains `predictions`, `logits`, `encoder_outputs`,
                `encoder_logits`, `encoder_output_lengths`.
        """
        logits = None
        inputs = self.embedding(inputs)
        encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)

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
            "encoder_output_lengths": encoder_output_lengths,
        }
    
    def training_step(self, batch: tuple) -> OrderedDict:
        r"""
        Forward propagate a `inputs` and `targets` pair for training.

        Inputs:
            train_batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`

        Returns:
            loss (torch.Tensor): loss for training
        """
        inputs, targets, input_lengths, target_lengths = batch

        inputs = self.embedding(inputs)
        encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
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

        inputs = self.embedding(inputs)
        encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        logits = self.decoder(
            encoder_outputs,
            encoder_output_lengths=encoder_output_lengths,
            teacher_forcing_ratio=0.0,
        )
        return self.collect_outputs(
            stage="val",
            logits=logits,
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

        inputs = self.embedding(inputs)
        encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        logits = self.decoder(
            encoder_outputs,
            encoder_output_lengths=encoder_output_lengths,
            teacher_forcing_ratio=0.0,
        )
        return self.collect_outputs(
            stage="test",
            logits=logits,
            encoder_output_lengths=encoder_output_lengths,
            targets=targets,
            target_lengths=target_lengths,
        )

      