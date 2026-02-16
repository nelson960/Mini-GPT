from __future__ import annotations

import json
import math
import os
import traceback
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import torch
from tqdm import tqdm

from .data import load_text, select_device
from .modern_model import ModernGPT, ModernGPTConfig
from .modern_tokenizer import (
    Tokenizer,
    build_tokenizer_from_text,
    tokenizer_to_payload,
)


@dataclass
class ModernTrainConfig:
    # data
    input_path: str = "data/input.txt"
    train_frac: float = 0.9
    tokenizer: str = "bpe"
    bpe_vocab_size: int = 256
    bpe_min_pair_freq: int = 2

    # model
    block_size: int = 128
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 192
    dropout: float = 0.1
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-8
    ffn_mult: float = 4.0

    # moe
    use_moe: bool = False
    n_experts: int = 4
    moe_top_k: int = 1
    moe_aux_loss_coef: float = 1e-2

    # optimization
    lr: float = 3e-4
    fixed_lr: Optional[float] = None
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    batch_size: int = 64
    max_steps: int = 2000

    # eval/logging
    eval_every: int = 200
    eval_iters: int = 30
    seed: int = 0
    patience: Optional[int] = None
    min_delta: float = 1e-4

    # checkpoints
    out_dir: Optional[str] = None


@dataclass
class ModernDataset:
    train_data: torch.Tensor
    val_data: torch.Tensor
    tokenizer: Tokenizer

    @classmethod
    def from_text(
        cls,
        text: str,
        train_frac: float = 0.9,
        tokenizer: str = "bpe",
        bpe_vocab_size: int = 256,
        bpe_min_pair_freq: int = 2,
    ) -> "ModernDataset":
        tok = build_tokenizer_from_text(
            text,
            tokenizer=tokenizer,
            bpe_vocab_size=bpe_vocab_size,
            bpe_min_pair_freq=bpe_min_pair_freq,
        )
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
        x = torch.stack([d[i : i + block_size] for i in ix])
        y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: ModernGPT,
    dataset: ModernDataset,
    cfg: ModernTrainConfig,
    device: torch.device,
    pbar: tqdm | None = None,
) -> Dict[str, float]:
    model.eval()
    out: Dict[str, float] = {}
    position = 1 if pbar is not None else None

    for split in ["train", "val"]:
        losses = []
        for _ in tqdm(range(cfg.eval_iters), desc=f"Evaluating {split}", leave=False, position=position):
            xb, yb = dataset.get_batch(split, cfg.batch_size, cfg.block_size, device)
            _, loss = model(xb, yb)
            losses.append(loss.detach().item())
        out[split] = sum(losses) / len(losses)

    model.train()
    return out


def save_checkpoint(
    path: str,
    model: ModernGPT,
    opt: torch.optim.Optimizer,
    tokenizer: Tokenizer,
    train_cfg: ModernTrainConfig,
    step: int,
    best_val: Optional[float] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "arch": "modern_gpt",
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
        "tokenizer": tokenizer_to_payload(tokenizer),
        "train_cfg": asdict(train_cfg),
        "model_cfg": asdict(model.cfg),
        "step": step,
        "best_val": best_val,
    }
    torch.save(payload, path)


def train_modern(train_cfg: ModernTrainConfig) -> None:
    torch.manual_seed(train_cfg.seed)
    device = select_device(prefer_mps=True)
    print("device:", device)

    text = load_text(train_cfg.input_path)
    dataset = ModernDataset.from_text(
        text,
        train_frac=train_cfg.train_frac,
        tokenizer=train_cfg.tokenizer,
        bpe_vocab_size=train_cfg.bpe_vocab_size,
        bpe_min_pair_freq=train_cfg.bpe_min_pair_freq,
    )

    model_cfg = ModernGPTConfig(
        vocab_size=dataset.tokenizer.vocab_size,
        block_size=train_cfg.block_size,
        n_layer=train_cfg.n_layer,
        n_head=train_cfg.n_head,
        n_embd=train_cfg.n_embd,
        dropout=train_cfg.dropout,
        rope_theta=train_cfg.rope_theta,
        rms_norm_eps=train_cfg.rms_norm_eps,
        ffn_mult=train_cfg.ffn_mult,
        use_moe=train_cfg.use_moe,
        n_experts=train_cfg.n_experts,
        moe_top_k=train_cfg.moe_top_k,
        moe_aux_loss_coef=train_cfg.moe_aux_loss_coef,
    )

    model = ModernGPT(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    warmup_steps = max(50, train_cfg.max_steps // 20)
    min_lr = train_cfg.lr * 0.1

    def lr_at(step: int) -> float:
        if train_cfg.fixed_lr is not None:
            return train_cfg.fixed_lr
        if step < warmup_steps:
            return train_cfg.lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, train_cfg.max_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr + (train_cfg.lr - min_lr) * cosine

    if train_cfg.out_dir is not None:
        os.makedirs(train_cfg.out_dir, exist_ok=True)

    best_val = float("inf")
    best_step = -1
    no_improve = 0
    train_loss = 0.0
    val_loss = float("inf")

    step = -1
    stop_reason = "completed"
    caught_exc: Exception | None = None
    pbar = tqdm(range(train_cfg.max_steps), desc="Training", position=0)
    try:
        for step in pbar:
            lr = lr_at(step)
            for pg in opt.param_groups:
                pg["lr"] = lr

            xb, yb = dataset.get_batch("train", train_cfg.batch_size, train_cfg.block_size, device)
            _, loss = model(xb, yb)
            train_loss = loss.detach().item()

            pbar.set_description(f"Training | loss: {train_loss:.4f} | best_val: {best_val:.4f}")

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            opt.step()

            if step % train_cfg.eval_every == 0 or step == train_cfg.max_steps - 1:
                losses = estimate_loss(model, dataset, train_cfg, device, pbar)
                train_loss, val_loss = losses["train"], losses["val"]
                pbar.write(f"step {step:4d} | lr {lr:.2e} | train {train_loss:.4f} | val {val_loss:.4f}")

                improved = val_loss < best_val - train_cfg.min_delta
                if improved:
                    best_val = val_loss
                    best_step = step
                    no_improve = 0
                    if train_cfg.out_dir is not None:
                        save_checkpoint(
                            os.path.join(train_cfg.out_dir, "best.pt"),
                            model,
                            opt,
                            dataset.tokenizer,
                            train_cfg,
                            step,
                            best_val=best_val,
                        )
                        pbar.write(f"*** New best model saved: val {val_loss:.4f} @ step {step} ***")
                else:
                    no_improve += 1

                if train_cfg.patience is not None and no_improve >= train_cfg.patience:
                    stop_reason = (
                        f"early_stopping_no_improve_{no_improve}_evals"
                    )
                    pbar.write(
                        f"early stopping: no improvement for {no_improve} evals "
                        f"({no_improve * train_cfg.eval_every} steps). "
                        f"best val {best_val:.4f} @ step {best_step}"
                    )
                    break
    except KeyboardInterrupt:
        stop_reason = "interrupted_keyboard"
        print("\ntraining interrupted by user (KeyboardInterrupt).")
    except Exception as exc:
        stop_reason = f"error_{type(exc).__name__}"
        caught_exc = exc
        print("\ntraining crashed. writing partial results before re-raising:")
        traceback.print_exc()
    finally:
        pbar.close()
        print(f"done. best val {best_val:.4f} @ step {best_step}")

    if train_cfg.out_dir is not None:
        print(f"checkpoint saved to: {train_cfg.out_dir}/best.pt")
        results_path = os.path.join(train_cfg.out_dir, "results.txt")
        with open(results_path, "w", encoding="utf-8") as f:
            f.write(f"best_val = {best_val:.4f}\n")
            f.write(f"best_step = {best_step}\n")
            f.write(f"final_train_loss = {train_loss:.4f}\n")
            f.write(f"final_val_loss = {val_loss:.4f}\n")
            f.write(f"total_steps = {max(step + 1, 0)}\n")
            f.write(f"stop_reason = {stop_reason}\n")
            f.write("\n--- Configuration ---\n")
            for key, value in asdict(train_cfg).items():
                if key != "out_dir":
                    f.write(f"{key} = {value}\n")
        print(f"results saved to: {results_path}")

        summary_path = os.path.join(train_cfg.out_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_val": best_val,
                    "best_step": best_step,
                    "final_train_loss": train_loss,
                    "final_val_loss": val_loss,
                    "total_steps": max(step + 1, 0),
                    "stop_reason": stop_reason,
                    "train_cfg": asdict(train_cfg),
                },
                f,
                indent=2,
            )
        print(f"summary saved to: {summary_path}")

    if caught_exc is not None:
        raise caught_exc
