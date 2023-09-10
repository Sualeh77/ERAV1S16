import torch
from dataclasses import dataclass


## Get Config
@dataclass
class Config:
    dataset_path = "./data"
    batch_size = 128
    num_epochs = 25
    lr = 2e-3 # 10**-4
    seq_len = 160
    d_model = 512
    lang_src = "en"
    lang_tgt = "fr"
    tokenizer_file = "tokenizer_{0}.json"
    num_workers = 4
    pin_memory = True
    weight_decay = 1e-4
    eps = 1e-9
    label_smoothing = 0.1
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"