from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from minigpt.train_modern import ModernTrainConfig, train_modern


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train a modern MiniGPT (RMSNorm + RoPE + SwiGLU + optional MoE) from scratch."
    )

    # data
    p.add_argument("--input", dest="input_path", default="data/input.txt", help="Path to training text")
    p.add_argument("--out_dir", default=None, help="Checkpoint directory")
    p.add_argument("--train_frac", type=float, default=0.9)
    p.add_argument(
        "--tokenizer",
        choices=["char", "bpe"],
        default="bpe",
        help="Tokenizer type for training text.",
    )
    p.add_argument("--bpe_vocab_size", type=int, default=256, help="Target vocab size for BPE tokenizer.")
    p.add_argument(
        "--bpe_min_pair_freq",
        type=int,
        default=2,
        help="Minimum pair frequency required to create a BPE merge.",
    )

    # model
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=192)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--rope_theta", type=float, default=10000.0)
    p.add_argument("--rms_norm_eps", type=float, default=1e-8)
    p.add_argument("--ffn_mult", type=float, default=4.0)

    # moe
    p.add_argument("--use_moe", action="store_true", help="Enable MoE in feed-forward layers")
    p.add_argument("--n_experts", type=int, default=4)
    p.add_argument("--moe_top_k", type=int, default=1)
    p.add_argument("--moe_aux_loss_coef", type=float, default=1e-2)

    # optimization
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--fixed_lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_steps", type=int, default=2000)

    # eval/logging
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--eval_iters", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--min_delta", type=float, default=1e-4)

    args = p.parse_args()

    cfg = ModernTrainConfig(
        input_path=args.input_path,
        out_dir=args.out_dir,
        train_frac=args.train_frac,
        tokenizer=args.tokenizer,
        bpe_vocab_size=args.bpe_vocab_size,
        bpe_min_pair_freq=args.bpe_min_pair_freq,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        rope_theta=args.rope_theta,
        rms_norm_eps=args.rms_norm_eps,
        ffn_mult=args.ffn_mult,
        use_moe=args.use_moe,
        n_experts=args.n_experts,
        moe_top_k=args.moe_top_k,
        moe_aux_loss_coef=args.moe_aux_loss_coef,
        lr=args.lr,
        fixed_lr=args.fixed_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        eval_iters=args.eval_iters,
        seed=args.seed,
        patience=args.patience,
        min_delta=args.min_delta,
    )

    train_modern(cfg)


if __name__ == "__main__":
    main()
