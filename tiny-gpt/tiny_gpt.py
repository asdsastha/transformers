import math
import time
from dataclasses import dataclass 
from pathlib import Path

import torch 
import torch.nn as nn
from torch.nn import functional as F
from model import GPT, GPTConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\ttorch: {torch.__version__}\n\tdevice: {device}({torch.cuda.get_device_name(0)})")

torch.manual_seed(13337) # set the random seed for reproducibility


# Extracting Data and Tokenizing
# Character-level "tokenizer": every distinct character gets an id. That gives a vocab of 65,
# versus ~50k for GPT-2's BPE. Simple, but each token carries little meaning, so the model
# has to learn spelling as well as language.
class GPTWrapper:
    def __init__(self, file_path):
        self.text = Path(file_path).read_text()
        self.vocab = sorted(list(set(self.text)))
        self.vocab_size = len(self.vocab)
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}
        self.data = torch.tensor([self.stoi[ch] for ch in self.text], dtype=torch.long)

        n = int(0.9 * len(self.data))  # 90% of the data for training, rest for validation
        self.split_data = {'train': self.data[:n], 'val': self.data[n:]}
        self.cfg = GPTConfig(vocab_size=self.vocab_size)
        self.model = GPT(self.cfg)

    def get_batch(self, split):
        data = self.split_data['train'] if split == 'train' else self.split_data['val']
        ix = torch.randint(len(data) - self.cfg.block_size, (self.cfg.batch_size,))
        x = torch.stack([data[i:i+self.cfg.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.cfg.block_size+1] for i in ix])
        return x, y


    def print(self):
        print(f"Text length: {len(self.text)}")
        print(f"Vocab size: {self.vocab_size}")
        print(f"Data shape: {self.data.shape}")
        print(f"Train data shape: {self.split_data['train'].shape}")
        print(f"Validation data shape: {self.split_data['val'].shape}")
        print(f"Vocab: {self.vocab}")
        print("\nToken mapping (int <-> string):")
        print(f"{'Index':>5} | {'Token':<8}")
        print("-" * 16)
        for idx, token in enumerate(self.vocab):
            printable_token = repr(token)
            print(f"{idx:>5} | {printable_token:<8}")

        print(f"Model: {self.model}")
        print(f"Config: {self.cfg}")


gptW = GPTWrapper("data/input.txt")
gptW.print()

