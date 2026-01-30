from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class CharTokenizer:
    """Character-level tokenizer.

    Builds a vocabulary from unique characters in a training text.
    Provides encode/decode and save/load so the mapping stays stable across runs.
    """

    chars: List[str]

    def __post_init__(self) -> None:
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(self.chars)}
        self.itos: Dict[int, str] = {i: ch for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(list(set(text)))
        return cls(chars=chars)

    def encode(self, s: str) -> List[int]:
        # Raises KeyError if unseen chars appear (intentional for learning/debugging).
        return [self.stoi[c] for c in s]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chars": self.chars}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return cls(chars=obj["chars"])
