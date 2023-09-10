import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super(BilingualDataset, self).__init__()
        self.seq_len = seq_len
        self.ds = ds

        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Use max of source and target lengths instead of max sequence length
        max_len = max(len(enc_input_tokens) + 2, len(dec_input_tokens) + 1)

        # Add sos, eos and padding to each sentence
        #enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2   # we will add <s> and </s> for sos and eos
        enc_num_padding_tokens = max_len - len(enc_input_tokens) - 2
        
        # We will only add <s> in input of decoder. </s> is added at labeling (so that model learns where sentence shud end).
        #dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1
        dec_num_padding_tokens = max_len - len(dec_input_tokens) - 1

        # Make sure the num of padding tokens is not -ve. If it is, the sentence is too long (longer than seq_len).
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")
        
        # Add <s> and </s> token | for encoder input
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                #torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only <s> token | For decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                #torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only </s> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                #torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        # Double check size of the tensors to make sure they are all seq_len long
        # assert encoder_input.size(0) == self.seq_len
        # assert decoder_input.size(0) == self.seq_len
        # assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,     # of size of seq_len
            "decoder_input": decoder_input,     # of size of seq_len
            #"encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),      # (1,1,seq_len)
            #"decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),      # (1, seq_len) & (1, seq_len, seq_len)
            "label": label,                     # of size of seq_len
            "src_text": src_text,
            "tgt_text": tgt_text,
            #"encoder_str_len": len(encoder_input),
            #"decoder_str_len": len(dec_input_tokens)
            "max_len": max_len
        }
    
    def custom_collate_fn(self, batch):
        """
        Pads the batches so that they all have the same length.

        Args:
            batch: The batch of data.

        Returns:
            The padded batch.
        """

        max_len = max(list(map(lambda x: x["max_len"], batch)))

        encoder_inputs, decoder_inputs, labels, src_text, tgt_text = [], [], [], [], []

        for item in batch:
            encoder_inputs.append(
                torch.cat(
                    [
                        item["encoder_input"],
                        torch.tensor([self.pad_token] * (max_len - item["encoder_input"].size(0)), dtype=torch.int64,),
                    ],
                    dim=0
                )
            )

            decoder_inputs.append(
                torch.cat(
                    [
                        item["decoder_input"],
                        torch.tensor([self.pad_token] * (max_len - item["decoder_input"].size(0)), dtype=torch.int64,),
                    ],
                    dim=0
                )
            )
            labels.append(
                torch.cat(
                    [
                        item["label"],
                        torch.tensor([self.pad_token] * (max_len - item["label"].size(0)),dtype=torch.int64,),
                    ],
                    dim=0
                )
            )
            src_text.append(item["src_text"])
            tgt_text.append(item["tgt_text"])

        # # Pad sequences within the batch to the length of the longest sequence
        # encoder_inputs_padded = pad_sequence(
        #     encoder_inputs, batch_first=True, padding_value=self.pad_token.item()
        # )
        # decoder_inputs_padded = pad_sequence(
        #     decoder_inputs, batch_first=True, padding_value=self.pad_token.item()
        # )
        encoder_inputs_padded = torch.stack(encoder_inputs)
        decoder_inputs_padded = torch.stack(decoder_inputs)

        # encoder_masks_padded = pad_sequence(
        #     encoder_masks, batch_first=True, padding_value=0
        # )

        # encoder_masks_padded = (
        #     (encoder_inputs_padded != self.pad_token).unsqueeze(1).unsqueeze(1).int()
        # )

        encoder_masks_padded = torch.vstack(
            [
                ((eip != self.pad_token).unsqueeze(0).unsqueeze(0).unsqueeze(0).int())
                for eip in encoder_inputs_padded
            ]
        )

        # decoder_masks_padded = pad_sequence(
        #     decoder_masks, batch_first=True, padding_value=0
        # )

        # decoder_masks_padded = (decoder_inputs_padded != self.pad_token).unsqueeze(
        #     1
        # ).unsqueeze(-1).int() & causal_mask_batch(
        #     batch_size, decoder_inputs_padded.size(1)
        # )  # (batch, 1, seq_len, 1) & (batch, 1, seq_len, seq_len)

        decoder_masks_padded = torch.vstack(
            [
                ((dip != self.pad_token).unsqueeze(0).int() & causal_mask(dip.size(0))).unsqueeze(0)
                for dip in decoder_inputs_padded
            ]
        )

        # labels_padded = pad_sequence(
        #     labels, batch_first=True, padding_value=self.pad_token.item()
        # )

        labels_padded = torch.stack(labels)

        return {
            "encoder_input": encoder_inputs_padded,
            "decoder_input": decoder_inputs_padded,
            "encoder_mask": encoder_masks_padded,
            "decoder_mask": decoder_masks_padded,
            "label": labels_padded,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)      # Creates Upper triangular metrics with values as 1
    return mask == 0