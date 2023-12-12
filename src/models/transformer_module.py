from collections import OrderedDict
from typing import Any, Dict

from omegaconf import DictConfig 

import torch
from torch import Tensor
import pytorch_lightning as pl

from src.models.transformer.model.encoder import Encoder
from src.models.transformer.model.decoder import Decoder
from src.data.components.vocab import Vocab
from src.models.transformer.modules.utils import get_class_name
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.text import BLEUScore

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
        encoder:    Encoder, 
        decoder:    Decoder, 
        pad_id:     int, 
        sos_id:     int,
        eos_id:     int,
        teacher_forcing_ratio: float,
        scheduler:  torch.optim.lr_scheduler,
        opimizer:   torch.optim.Optimizer,
        compile:    bool, 
    ) -> None:
        super(TransformerLitModule, self).__init__()
        self.save_hyperparameters(logger=False)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id)
        
        self.pad_id=pad_id
        self.sos_id=sos_id
        self.eos_id=eos_id

        self.vocab = None
        self.encoder = encoder
        self.decoder = decoder
        
        # metric objects for calculating and averaging accuracy across batches
        self.train_bleu = BLEUScore()
        self.val_bleu = BLEUScore()
        self.test_bleu = BLEUScore()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_bleu_best = MaxMetric()
    
    def one_hot_vector(self, ):
        
    def load_vocab(self, vocab: Vocab):
        self.vocab = vocab
    
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_bleu.reset()
        self.val_bleu_best.reset()

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
        inputs, targets, input_lengths, target_lengths = batch["inputs"], batch["targets"], batch["inputs_length"], batch["targets_length"]

        inputs = self.vocab.embed(inputs)
        encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        if get_class_name(self.decoder) == "Decoder":
            logits = self.decoder(
                encoder_outputs=encoder_outputs,
                targets=targets,
                encoder_output_lengths=encoder_output_lengths,
                target_lengths=target_lengths,
                teacher_forcing_ratio=self.hparams.teacher_forcing_ratio,
            )
        else:
            raise ValueError("Why is your decoder not a Decoder?")
        
        loss, predictions, _ =  self.collect_outputs(   stage="train",
                                                        logits=logits,
                                                        targets=targets,
                                                        target_lengths=target_lengths,)
        
        
        self.train_loss(loss)
        self.train_bleu(predictions, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/bleu", self.train_bleu, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: tuple, batch_idx: int) -> OrderedDict:
        r"""
        Forward propagate a `inputs` and `targets` pair for validation.

        Inputs:
            train_batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.Tensor): loss for training
        """
        inputs, targets, input_lengths, target_lengths = batch["inputs"], batch["targets"], batch["inputs_length"], batch["targets_length"]

        inputs = self.vocab.embed(inputs)
        encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        logits = self.decoder(
            encoder_outputs,
            encoder_output_lengths=encoder_output_lengths,
            teacher_forcing_ratio=0.0,
        )
        
        loss, predictions, _ =  self.collect_outputs(stage="val",
                                                    logits=logits,
                                                    encoder_output_lengths=encoder_output_lengths,
                                                    targets=targets,
                                                    target_lengths=target_lengths,)
        
        self.val_loss(loss)
        self.val_bleu(predictions, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/bleu", self.val_bleu, on_step=False, on_epoch=True, prog_bar=True)
    
    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
        
    def on_validation_epoch_end(self) -> None:
        self.val_bleu.compute()
        self.val_bleu_best(self.val_bleu)
        self.log("val/bleu_best", self.val_bleu_best.compute(), sync_dist=True, prog_bar=True)
        return super().on_validation_epoch_end()
    
    def test_step(self, batch: tuple, batch_idx: int) -> OrderedDict:
        r"""
        Forward propagate a `inputs` and `targets` pair for test.

        Inputs:
            train_batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.Tensor): loss for training
        """
        inputs, targets, input_lengths, target_lengths = batch["inputs"], batch["targets"], batch["inputs_length"], batch["targets_length"]

        inputs = self.vocab.embed(inputs)
        encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        logits = self.decoder(
            encoder_outputs,
            encoder_output_lengths=encoder_output_lengths,
            teacher_forcing_ratio=0.0,
        )
        loss, predictions, _ = self.collect_outputs(stage="test",
                                                    logits=logits,
                                                    encoder_output_lengths=encoder_output_lengths,
                                                    targets=targets,
                                                    target_lengths=target_lengths,)
        self.test_loss(loss)
        self.test_bleu(predictions, targets)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}