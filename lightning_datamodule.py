import os

from dataset import BilingualDataset
from torch.utils.data import Dataset, DataLoader, random_split


# Huggingface datasets and tokenizers
from datasets import load_dataset, Dataset as HfDataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from typing import Any, List, Optional, Union
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
import albumentations as A

from torch.utils.data import ConcatDataset

class BilingualDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path: str, src_lang: str, tgt_lang: str, seq_len: int = 350, batch_size: int = 512,
                num_workers: int = 0, pin_memory: bool = False, tokenizer_file: str = "tokenizer_{0}.json",
    ):
        super(BilingualDataModule, self).__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.src_tokenizer_path = os.path.join(dataset_path, tokenizer_file.format(src_lang))
        self.tgt_tokenizer_path = os.path.join(dataset_path, tokenizer_file.format(tgt_lang))

        self.train_ds_raw = None
        self.val_ds_raw = None

        self.tokenizer_src: Tokenizer = None
        self.tokenizer_tgt: Tokenizer = None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

        raw_ds: HfDataset = load_dataset("opus_books", f"{self.hparams.src_lang}-{self.hparams.tgt_lang}", split="train",
                                         cache_dir=self.hparams.dataset_path,)
        
        # Build tokenizers
        self.tokenizer_src = self.get_or_build_tokenizer(self.src_tokenizer_path, raw_ds, self.hparams.src_lang)
        self.tokenizer_tgt = self.get_or_build_tokenizer(self.tgt_tokenizer_path, raw_ds, self.hparams.tgt_lang)

        # Add Filtering
        ## 1. Remove all samples with length more than 150
        ## 2. Remove all samples with length of target more than 10 of source
        raw_ds = raw_ds.filter(lambda x: len(self.tokenizer_src.encode(x["translation"][self.hparams.src_lang]).ids) <= 150)
        raw_ds = raw_ds.filter(lambda x: len(self.tokenizer_tgt.encode(x["translation"][self.hparams.tgt_lang]).ids) <= len(self.tokenizer_src.encode(x["translation"][self.hparams.src_lang]).ids) + 10)

        # Keep 90% for training, 10% for validation
        self.train_ds_raw, self.val_ds_raw = random_split(raw_ds, [0.9, 0.1])

        # Find the maximum length of each sentence in the source and target sentence
        max_len_src = 0
        max_len_tgt = 0

        for item in raw_ds:
            src_ids = self.tokenizer_src.encode(item["translation"][self.hparams.src_lang]).ids
            tgt_ids = self.tokenizer_tgt.encode(item["translation"][self.hparams.tgt_lang]).ids
            max_len_src = max(max_len_src, len(src_ids))
            max_len_tgt = max(max_len_tgt, len(tgt_ids))

        print(f"Max length of source sentence: {max_len_src}")
        print(f"Max length of target sentence: {max_len_tgt}")

    def get_all_sentences(self, raw_ds, lang):
        for item in raw_ds:
            yield item["translation"][lang]

    def get_or_build_tokenizer(self, tokenizer_path, raw_ds, lang):
        if not os.path.exists(tokenizer_path):
            # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
            tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = WordLevelTrainer(
                special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
            )
            tokenizer.train_from_iterator(
                self.get_all_sentences(raw_ds, lang), trainer=trainer
            )
            tokenizer.save(str(tokenizer_path))
        else:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))

        return tokenizer
    
    def setup(self, stage="fit"):
        self.data_train = BilingualDataset(
            self.train_ds_raw,
            self.tokenizer_src,
            self.tokenizer_tgt,
            self.hparams.src_lang,
            self.hparams.tgt_lang,
            self.hparams.seq_len,
        )

        self.data_val = BilingualDataset(
            self.val_ds_raw,
            self.tokenizer_src,
            self.tokenizer_tgt,
            self.hparams.src_lang,
            self.hparams.tgt_lang,
            self.hparams.seq_len,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=False,
            collate_fn=self.data_train.custom_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=False,
            collate_fn=self.data_val.custom_collate_fn
        )
