import torch
import torchmetrics
import pytorch_lightning as pl
import torch.nn as nn

from torch.optim.lr_scheduler import OneCycleLR
from tokenizers import Tokenizer
from model import build_transformer, Transformer
from greedy_decoder import greedy_decode

class LightningTransformer(pl.LightningModule):
    def __init__(self, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer, seq_len: int = 350, d_model: int = 512,
                 lr=1e-3, weight_decay=1e-4, eps=1e-9, label_smoothing=0.1,
    ):
        super(LightningTransformer, self).__init__()

        self.tokenizer_tgt = tokenizer_tgt

        self.src_vocab_size = tokenizer_src.get_vocab_size()
        self.tgt_vocab_size = tokenizer_tgt.get_vocab_size()

        self.tgt_sos_idx = tokenizer_tgt.token_to_id("[SOS]")
        self.tgt_eos_idx = tokenizer_tgt.token_to_id("[EOS]")

        self.seq_len = seq_len

        self.model = build_transformer(
            self.src_vocab_size,
            self.tgt_vocab_size,
            seq_len,
            seq_len,
            d_model=d_model,
            d_ff=256,
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps

        self.criteria = nn.CrossEntropyLoss(
            ignore_index=tokenizer_src.token_to_id("[PAD]"),
            label_smoothing=label_smoothing,
        )

        self.training_step_outputs = []
        self.validation_step_outputs = {"source": [], "target": [], "predicted": []}
        self.predict_step_outputs = []

    def forward(self, batch):
        encoder_input = batch["encoder_input"]  # (b, seq_len)
        decoder_input = batch["decoder_input"]  # (B. sea len)
        encoder_mask = batch["encoder_mask"]  # (B, 1, 1, seq_len)
        decoder_mask = batch["decoder_mask"]  # (B, 1, seq_len, seq len)

        # Run the tensors through the encoder, decoder and the projection: layer

        encoder_output = self.model.encode(
            encoder_input, encoder_mask
        )  # (B, seq_len, d_model)
        decoder_output = self.model.decode(
            encoder_output, encoder_mask, decoder_input, decoder_mask
        )  #
        proj_output = self.model.project(decoder_output)  # (B, seq_len, vocab_size)

        return proj_output
    
    def calculate_loss(self, output, target):
        loss = self.criteria(output.view(-1, self.tgt_vocab_size), target.view(-1))
        return loss

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)

        # self.training_step_outputs.append(self.check_class_accuracy(output, target))

        loss = self.calculate_loss(output, batch["label"])

        self.log_dict(
            {
                "train_loss": loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss
    
    def validation_step(self, batch, batch_idx):
        encoder_input = batch["encoder_input"]  # (b, seq_len)
        encoder_mask = batch["encoder_mask"]  # (b, 1, 1, seq_len)

        out = greedy_decode(
            self.model,
            encoder_input,
            encoder_mask,
            self.tgt_sos_idx,
            self.tgt_eos_idx,
            self.seq_len,
            self.device,
        )

        source_text = batch["src_text"][0]
        target_text = batch["tgt_text"][0]
        out_text = self.tokenizer_tgt.decode(out.detach().cpu().numpy())

        self.validation_step_outputs["source"].append(source_text)
        self.validation_step_outputs["target"].append(target_text)
        self.validation_step_outputs["predicted"].append(out_text)

        # loss = self.calculate_loss(output, target, S)

    def on_validation_epoch_end(self):
        self.log_dict(
            {
                "val_cer": torchmetrics.text.CharErrorRate()(
                    self.validation_step_outputs["predicted"],
                    self.validation_step_outputs["target"],
                ),
                "val_wer": torchmetrics.text.WordErrorRate()(
                    self.validation_step_outputs["predicted"],
                    self.validation_step_outputs["target"],
                ),
                "val_bleu": torchmetrics.text.BLEUScore()(
                    self.validation_step_outputs["predicted"],
                    self.validation_step_outputs["target"],
                ),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.validation_step_outputs.clear()  # free memory
        self.validation_step_outputs = {"source": [], "target": [], "predicted": []}


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, eps=self.eps
        )  # weight_decay=self.weight_decay

        num_epochs = self.trainer.max_epochs
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            epochs=num_epochs,
            pct_start=1 / 10 if num_epochs != 1 else 0.5,
            div_factor=10,
            three_phase=True,
            final_div_factor=10,
            anneal_strategy="linear",
        )

        return (
            {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
                },
            },
        )
