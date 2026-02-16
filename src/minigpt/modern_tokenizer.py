from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

from .tokenizer import CharTokenizer


def _merge_once(ids: List[int], a: int, b: int, new_id: int) -> List[int]:
    out: List[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i + 1 < n and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


@dataclass
class BPETokenizer:
    """Sequence-level BPE tokenizer trained from raw text."""

    pieces: List[str]
    merges: List[Tuple[int, int]]
    base_vocab_size: int

    def __post_init__(self) -> None:
        if self.base_vocab_size < 1:
            raise ValueError("base_vocab_size must be >= 1")
        if self.base_vocab_size > len(self.pieces):
            raise ValueError("base_vocab_size cannot exceed total number of pieces")
        self.stoi: Dict[str, int] = {p: i for i, p in enumerate(self.pieces)}
        self.itos: Dict[int, str] = {i: p for i, p in enumerate(self.pieces)}
        self.base_stoi: Dict[str, int] = {p: i for i, p in enumerate(self.pieces[: self.base_vocab_size])}

    @property
    def vocab_size(self) -> int:
        return len(self.pieces)

    @classmethod
    def from_text(
        cls,
        text: str,
        vocab_size: int = 256,
        min_pair_freq: int = 2,
    ) -> "BPETokenizer":
        if vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")
        if min_pair_freq < 2:
            raise ValueError("min_pair_freq must be >= 2")

        base_chars = sorted(list(set(text)))
        if not base_chars:
            raise ValueError("Cannot train tokenizer on empty text")
        if vocab_size < len(base_chars):
            raise ValueError(
                f"vocab_size={vocab_size} is smaller than number of unique chars={len(base_chars)}"
            )

        pieces = list(base_chars)
        stoi = {ch: i for i, ch in enumerate(pieces)}
        ids = [stoi[ch] for ch in text]
        merges: List[Tuple[int, int]] = []

        while len(pieces) < vocab_size:
            if len(ids) < 2:
                break

            pair_counts: Dict[Tuple[int, int], int] = {}
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
            if not pair_counts:
                break

            ranked = sorted(pair_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            chosen: Tuple[int, int] | None = None
            for (a, b), freq in ranked:
                if freq < min_pair_freq:
                    break
                merged_piece = pieces[a] + pieces[b]
                if merged_piece in stoi:
                    continue
                chosen = (a, b)
                break

            if chosen is None:
                break

            a, b = chosen
            new_id = len(pieces)
            merged_piece = pieces[a] + pieces[b]
            pieces.append(merged_piece)
            stoi[merged_piece] = new_id
            merges.append((a, b))
            ids = _merge_once(ids, a, b, new_id)

        return cls(pieces=pieces, merges=merges, base_vocab_size=len(base_chars))

    def encode(self, s: str) -> List[int]:
        missing = [c for c in sorted(set(s)) if c not in self.base_stoi]
        if missing:
            preview = "".join(missing[:10])
            raise KeyError(f"Prompt contains chars unseen in tokenizer vocabulary: {preview!r}")
        ids = [self.base_stoi[c] for c in s]
        for merge_idx, (a, b) in enumerate(self.merges):
            new_id = self.base_vocab_size + merge_idx
            ids = _merge_once(ids, a, b, new_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def to_payload(self) -> Dict[str, object]:
        return {
            "type": "bpe",
            "pieces": self.pieces,
            "base_vocab_size": self.base_vocab_size,
            "merges": [[a, b] for a, b in self.merges],
        }

    @classmethod
    def from_payload(cls, obj: Dict[str, object]) -> "BPETokenizer":
        pieces = obj.get("pieces")
        base_vocab_size = obj.get("base_vocab_size")
        merges_raw = obj.get("merges")
        if not isinstance(pieces, list):
            raise ValueError("Invalid BPE tokenizer payload: expected list under 'pieces'")
        if not isinstance(base_vocab_size, int):
            raise ValueError("Invalid BPE tokenizer payload: expected int under 'base_vocab_size'")
        if not isinstance(merges_raw, list):
            raise ValueError("Invalid BPE tokenizer payload: expected list under 'merges'")

        merges: List[Tuple[int, int]] = []
        for item in merges_raw:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError("Invalid merge pair in BPE tokenizer payload")
            a, b = item
            if not isinstance(a, int) or not isinstance(b, int):
                raise ValueError("Invalid merge ids in BPE tokenizer payload")
            merges.append((a, b))
        return cls(pieces=pieces, merges=merges, base_vocab_size=base_vocab_size)


Tokenizer = Union[CharTokenizer, BPETokenizer]


def build_tokenizer_from_text(
    text: str,
    tokenizer: str = "bpe",
    bpe_vocab_size: int = 256,
    bpe_min_pair_freq: int = 2,
) -> Tokenizer:
    kind = tokenizer.lower().strip()
    if kind == "char":
        return CharTokenizer.from_text(text)
    if kind == "bpe":
        return BPETokenizer.from_text(
            text,
            vocab_size=bpe_vocab_size,
            min_pair_freq=bpe_min_pair_freq,
        )
    raise ValueError(f"Unsupported tokenizer '{tokenizer}'. Use 'char' or 'bpe'.")


def tokenizer_to_payload(tokenizer: Tokenizer) -> Dict[str, object]:
    if isinstance(tokenizer, CharTokenizer):
        return {"type": "char", "chars": tokenizer.chars}
    if isinstance(tokenizer, BPETokenizer):
        return tokenizer.to_payload()
    raise TypeError(f"Unsupported tokenizer type: {type(tokenizer)!r}")


def tokenizer_from_payload(payload: Dict[str, object]) -> Tokenizer:
    tok_type = payload.get("type")
    # Backward compatibility with old char checkpoints.
    if tok_type is None and "chars" in payload:
        return CharTokenizer(chars=payload["chars"])
    if tok_type == "char":
        chars = payload.get("chars")
        if not isinstance(chars, list):
            raise ValueError("Invalid char tokenizer payload")
        return CharTokenizer(chars=chars)
    if tok_type == "bpe":
        return BPETokenizer.from_payload(payload)
    raise ValueError("Unsupported tokenizer payload in checkpoint")
