from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path so we can import minigpt
sys.path.insert(0, str(Path(__file__).parent / "src"))

from minigpt.train import TrainConfig, train


def main():
    p = argparse.ArgumentParser(description="Train a char-level MiniGPT next-token predictor.")
    p.add_argument("--input", dest="input_path", default="data/input.txt", help="Path to training text.")
    p.add_argument("--out_dir", default=None, help="Checkpoint directory (if not provided, models won't be saved).")
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--fixed_lr", type=float, default=None, help="Fixed learning rate (overrides scheduler)")
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--eval_iters", type=int, default=30)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=None, help="Early stopping: stop if no val improvement for N evals")
    p.add_argument("--min_delta", type=float, default=1e-4, help="Min change in val loss to count as improvement")
    args = p.parse_args()

    cfg = TrainConfig(
        input_path=args.input_path,
        out_dir=args.out_dir,
        block_size=args.block_size,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lr=args.lr,
        fixed_lr=args.fixed_lr,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        eval_every=args.eval_every,
        eval_iters=args.eval_iters,
        grad_clip=args.grad_clip,
        seed=args.seed,
        patience=args.patience,
        min_delta=args.min_delta,
    )
    train(cfg)


if __name__ == "__main__":
    main()
