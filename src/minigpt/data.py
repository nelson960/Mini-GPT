from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import torch

from .tokenizer import CharTokenizer


def select_device(prefer_mps: bool = True) -> torch.device:
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_text(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Provide a training text file, e.g. data/input.txt"
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@dataclass
class Dataset:
    """Holds tokenized data + provides get_batch."""

    train_data: torch.Tensor  # (N_train,)
    val_data: torch.Tensor    # (N_val,)
    tokenizer: CharTokenizer

    @classmethod
    def from_text(
        cls,
        text: str,
        train_frac: float = 0.9,
    ) -> "Dataset":
        tok = CharTokenizer.from_text(text)
        data = torch.tensor(tok.encode(text), dtype=torch.long)
        n = int(train_frac * len(data))
        return cls(train_data=data[:n], val_data=data[n:], tokenizer=tok)

    def get_batch(
        self,
        split: str,
        batch_size: int,
        block_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d = self.train_data if split == "train" else self.val_data
        if len(d) <= block_size + 1:
            raise ValueError(
                f"Data split too small for block_size={block_size}. Need > block_size+1 tokens."
            )

        ix = torch.randint(0, len(d) - block_size - 1, (batch_size,))
        x = torch.stack([d[i : i + block_size] for i in ix])                 # (B,T)
        y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])         # (B,T)
        return x.to(device), y.to(device)
