from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .data import Dataset, load_text, select_device
from .model import GPTConfig, MiniGPT
from .tokenizer import CharTokenizer


@dataclass
class TrainConfig:
    # data
    input_path: str = "data/input.txt"
    train_frac: float = 0.9

    # model
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1

    # optimization
    lr: float = 3e-4
    fixed_lr: Optional[float] = None  # if set, overrides lr scheduler
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    batch_size: int = 64
    max_steps: int = 2000

    # eval/logging
    eval_every: int = 200
    eval_iters: int = 30
    seed: int = 0
    patience: Optional[int] = None  # early stopping: stop if no improvement for N evals
    min_delta: float = 1e-4  # minimum change in val loss to qualify as improvement

    # checkpoints
    out_dir: Optional[str] = None


@torch.no_grad()
def estimate_loss(
    model: MiniGPT,
    dataset: Dataset,
    cfg: TrainConfig,
    device: torch.device,
    pbar: tqdm | None = None,
) -> Dict[str, float]:
    model.eval()
    out: Dict[str, float] = {}
    position = 1 if pbar is not None else None
    for split in ["train", "val"]:
        losses = []
        for _ in tqdm(
            range(cfg.eval_iters),
            desc=f"Evaluating {split}",
            leave=False,
            position=position,
        ):
            xb, yb = dataset.get_batch(split, cfg.batch_size, cfg.block_size, device)
            _, loss = model(xb, yb)
            losses.append(float(loss))
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def save_checkpoint(
    path: str,
    model: MiniGPT,
    opt: torch.optim.Optimizer,
    tokenizer: CharTokenizer,
    train_cfg: TrainConfig,
    step: int,
    best_val: Optional[float] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
        "tokenizer": {"chars": tokenizer.chars},
        "train_cfg": asdict(train_cfg),
        "gpt_cfg": asdict(model.cfg),
        "step": step,
        "best_val": best_val,
    }
    torch.save(payload, path)


def train(train_cfg: TrainConfig) -> None:
    torch.manual_seed(train_cfg.seed)
    device = select_device(prefer_mps=True)
    print("device:", device)

    text = load_text(train_cfg.input_path)
    dataset = Dataset.from_text(text, train_frac=train_cfg.train_frac)

    gpt_cfg = GPTConfig(
        vocab_size=dataset.tokenizer.vocab_size,
        block_size=train_cfg.block_size,
        n_layer=train_cfg.n_layer,
        n_head=train_cfg.n_head,
        n_embd=train_cfg.n_embd,
        dropout=train_cfg.dropout,
    )

    model = MiniGPT(gpt_cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )

    warmup_steps = 200
    min_lr = train_cfg.lr * 0.1

    def lr_at(step):
        # fixed lr override
        if train_cfg.fixed_lr is not None:
            return train_cfg.fixed_lr
        # warmup
        if step < warmup_steps:
            return train_cfg.lr * (step + 1) / warmup_steps
        # cosine decay
        progress = (step - warmup_steps) / max(1, (train_cfg.max_steps - warmup_steps))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr + (train_cfg.lr - min_lr) * cosine

    if train_cfg.out_dir is not None:
        os.makedirs(train_cfg.out_dir, exist_ok=True)

    best_val = float("inf")
    best_step = -1
    no_improve = 0  # count eval checkpoints without improvement
    train_loss = 0.0

    pbar = tqdm(range(train_cfg.max_steps), desc="Training", position=0)
    for step in pbar:
        lr = lr_at(step)
        for pg in opt.param_groups:
            pg["lr"] = lr

        xb, yb = dataset.get_batch("train", train_cfg.batch_size, train_cfg.block_size, device)
        _, loss = model(xb, yb)
        train_loss = loss.item()

        pbar.set_description(f"Training | loss: {train_loss:.4f} | best_val: {best_val:.4f}")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        opt.step()

        if step % train_cfg.eval_every == 0 or step == train_cfg.max_steps - 1:
            losses = estimate_loss(model, dataset, train_cfg, device, pbar)
            train_loss, val_loss = losses["train"], losses["val"]
            pbar.write(f"step {step:4d} | lr {lr:.2e} | train {train_loss:.4f} | val {val_loss:.4f}")

            # Check for improvement with min_delta threshold
            improved = val_loss < best_val - train_cfg.min_delta
            if improved:
                best_val = val_loss
                best_step = step
                no_improve = 0
                if train_cfg.out_dir is not None:
                    save_checkpoint(
                        os.path.join(train_cfg.out_dir, "best.pt"),
                        model, opt, dataset.tokenizer, train_cfg, step, best_val=best_val
                    )
                    pbar.write(f"*** New best model saved: val {val_loss:.4f} @ step {step} ***")
            else:
                no_improve += 1

            # Early stopping check
            if train_cfg.patience is not None and no_improve >= train_cfg.patience:
                pbar.write(
                    f"early stopping: no improvement for {no_improve} evals "
                    f"({no_improve * train_cfg.eval_every} steps). "
                    f"best val {best_val:.4f} @ step {best_step}"
                )
                break

    pbar.close()
    print(f"done. best val {best_val:.4f} @ step {best_step}")
    if train_cfg.out_dir is not None:
        print(f"checkpoint saved to: {train_cfg.out_dir}/best.pt")

        # Save results to results.txt
        results_path = os.path.join(train_cfg.out_dir, "results.txt")
        with open(results_path, "w") as f:
            f.write(f"best_val = {best_val:.4f}\n")
            f.write(f"best_step = {best_step}\n")
            f.write(f"final_train_loss = {train_loss:.4f}\n")
            f.write(f"final_val_loss = {val_loss:.4f}\n")
            f.write(f"total_steps = {step + 1}\n")
            f.write("\n--- Configuration ---\n")
            for key, value in asdict(train_cfg).items():
                if key != "out_dir":  # skip out_dir to avoid redundant path info
                    f.write(f"{key} = {value}\n")
        print(f"results saved to: {results_path}")
