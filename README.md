# MiniGPT Journey: Classic Transformer to Modern MoE (From Scratch)

This project tracks a full training journey on local hardware, starting from a classic GPT-style decoder and iterating to a modernized architecture with MoE.

## Constraints

- Data: only `data/input.txt`
- Training: from scratch (no pretrained weights)
- Hardware: Apple Silicon MacBook (MPS), 16GB unified memory
- Objective: minimize validation next-token cross-entropy
- Split: fixed contiguous `90% train / 10% val`
- Primary comparison mode: char-level tokenization

## MiniGPT (OG)

### MiniGPT (Char-level) — Next-token Predictor in PyTorch (MPS friendly)

A small, readable decoder-only Transformer that learns next-character prediction end-to-end.

End-to-end ML pipeline: data → tokenization → batching → training → evaluation → generation

```text
input.txt → encode chars → batches (x, y=x shifted) → MiniGPT → logits → cross-entropy → AdamW update
                                         ↑
                                   causal mask
```

- Understanding of logits / softmax / cross-entropy, causal masking, and train vs eval behavior
- Practical training hygiene: gradient clipping, periodic evaluation, checkpointing
- Hyperparameter tuning and regularization to combat overfitting
- Early stopping with patience-based validation monitoring
- Simple CLI tooling (`train.py`, `generate.py`) and reproducible configuration

## MiniGPT (Modern)

### MiniGPT Modern (RMSNorm + RoPE + SwiGLU + optional MoE/BPE)

A small, readable decoder-only Transformer that modernizes the block design while staying easy to train locally from scratch.

End-to-end ML pipeline: data → tokenization → batching → training → evaluation → generation

```text
input.txt → tokenize (char or BPE) → batches (x, y=x shifted) → ModernGPT → logits (+ MoE aux loss) → AdamW update
                                               ↑
                                         causal mask + RoPE
```

- Modernized internals: RMSNorm, RoPE, SwiGLU, optional MoE routing
- Better optimization and sequence handling under the same local compute constraints
- Same practical training hygiene: gradient clipping, periodic evaluation, checkpointing, early stopping
- Separate CLI tooling for modern experiments (`train_modern.py`, `generate_modern.py`)

## Metrics Definition + Fairness Notes

- `best_val`: lowest validation cross-entropy observed during training.
- Validation is computed by averaging `eval_iters` random validation batches of length `block_size`.
- Unit: nats per token (`-log p(token)`); lower is better.
- `best_step`: step where `best_val` was achieved and `best.pt` was saved.
- `final_val_loss`: validation loss at stop time; it can be worse than `best_val` due to overfitting after the best checkpoint.

Fairness notes:

- Char vs char runs are directly comparable by `best_val`.
- BPE vs char is not directly comparable by raw token loss because token granularity differs.
- For tokenizer-fair comparison, convert to a character-normalized metric (`nats/char`) before claiming one tokenizer is better.

## Architecture Evolution

| Stage                   | What changed                                           | Why                                              |
| ----------------------- | ------------------------------------------------------ | ------------------------------------------------ |
| Classic baseline        | Decoder-only Transformer, char tokenizer               | Establish reference performance                  |
| Better training hygiene | Stronger regularization + better eval + early stopping | Reduce overfitting under small data              |
| Modern dense            | RMSNorm + RoPE + SwiGLU                                | Better optimization and sequence handling        |
| Modern sparse           | MoE FFN (top-k routing + aux loss)                     | Improve quality under memory/compute constraints |
| Tokenizer experiment    | Optional BPE in modern path                            | Test efficiency/quality tradeoff                 |

## Run History (How We Reached the Best Score)

| Rank | Phase                |   best_val | Key change                                                     |
| ---: | -------------------- | ---------: | -------------------------------------------------------------- |
|    1 | Modern MoE best      | **1.8729** | Lower LR + higher WD + slightly stronger aux loss              |
|    2 | Modern MoE           |     1.8731 | Longer context (`block_size=192`)                              |
|    3 | Modern MoE           |     1.8748 | Better regularization/LR balance                               |
|    4 | Modern MoE           |     1.8848 | MoE routing outperformed dense modern control                  |
|    5 | Modern dense         |     1.8943 | RMSNorm/RoPE/SwiGLU gain over classic baseline                 |
|    6 | Modern dense control |     1.8966 | Dense modern reference near best dense                         |
|    7 | Classic best         |     1.9873 | More eval iters + tighter early-stop threshold                 |
|    8 | Classic tuned        |     1.9876 | Higher dropout + weight decay                                  |
|    9 | Classic baseline     |     2.0051 | First stable baseline                                          |
|   10 | BPE improved         |     3.0743 | Better than first BPE attempt, but still weaker than char mode |
|   11 | BPE test             |     4.4599 | Initial BPE setup underperformed on this tiny corpus           |

## Best Results Snapshot

| Track                            | Best run             |   best_val | best_step |
| -------------------------------- | -------------------- | ---------: | --------: |
| Classic (legacy loop)            | `p2_refine`          |     1.9873 |      3200 |
| Modern (RMSNorm/RoPE/SwiGLU/MoE) | `modern_moe_char_02` | **1.8729** |      1900 |

## Detailed Best-Config Comparison (Full Parameters)

| Parameter         | Classic Best (`p2_refine`) | Modern Best (`modern_moe_char_02`)                     |
| ----------------- | -------------------------- | ------------------------------------------------------ |
| checkpoint        | `runs/p2_refine/best.pt`   | `runs/modern_moe_char_02/best.pt`                      |
| input_path        | `data/input.txt`           | `data/input.txt`                                       |
| train_frac        | 0.9                        | 0.9                                                    |
| tokenizer         | char (implicit legacy)     | `char`                                                 |
| bpe_vocab_size    | n/a                        | 256 (unused in char mode)                              |
| bpe_min_pair_freq | n/a                        | 2 (unused in char mode)                                |
| block_size        | 128                        | 192                                                    |
| n_layer           | 4                          | 4                                                      |
| n_head            | 4                          | 4                                                      |
| n_embd            | 128                        | 128                                                    |
| dropout           | 0.20                       | 0.24                                                   |
| rope_theta        | n/a                        | 10000.0                                                |
| rms_norm_eps      | n/a                        | 1e-8                                                   |
| ffn_mult          | n/a                        | 2.0                                                    |
| use_moe           | no                         | yes                                                    |
| n_experts         | n/a                        | 4                                                      |
| moe_top_k         | n/a                        | 1                                                      |
| moe_aux_loss_coef | n/a                        | 0.0015                                                 |
| lr                | 2e-4                       | 9e-5                                                   |
| fixed_lr          | None                       | None                                                   |
| weight_decay      | 0.05                       | 0.08                                                   |
| grad_clip         | 1.0                        | 1.0                                                    |
| batch_size        | 64                         | 48                                                     |
| max_steps         | 50000                      | 5000                                                   |
| eval_every        | 200                        | 100                                                    |
| eval_iters        | 80                         | 100                                                    |
| seed              | 1337                       | 1337                                                   |
| patience          | 20                         | 8                                                      |
| min_delta         | 0.0002                     | 0.0002                                                 |
| best_val          | 1.9873                     | **1.8729**                                             |
| best_step         | 3200                       | 1900                                                   |
| final_train_loss  | 0.1443                     | 1.5611                                                 |
| final_val_loss    | 3.0598                     | 1.8729                                                 |
| total_steps       | 7201                       | 2001                                                   |
| stop_reason       | early stopping             | `interrupted_keyboard` (best checkpoint already saved) |

## Reproduce Best Runs

### Classic best (`p2_refine`)

```bash
python3 train.py --input data/input.txt --out_dir runs/p2_refine \
  --block_size 128 --batch_size 64 --max_steps 50000 --lr 2e-4 \
  --n_layer 4 --n_head 4 --n_embd 128 --dropout 0.20 --weight_decay 0.05 \
  --eval_every 200 --eval_iters 80 --grad_clip 1.0 --seed 1337 \
  --patience 20 --min_delta 0.0002
```

### Modern best (`modern_moe_char_02`)

```bash
python3 train_modern.py --input data/input.txt --out_dir runs/modern_moe_char_02 \
  --tokenizer char \
  --block_size 192 --n_layer 4 --n_head 4 --n_embd 128 --dropout 0.24 \
  --ffn_mult 2 --use_moe --n_experts 4 --moe_top_k 1 --moe_aux_loss_coef 0.0015 \
  --lr 9e-5 --weight_decay 0.08 --batch_size 48 --max_steps 5000 \
  --eval_every 100 --eval_iters 100 --patience 8 --min_delta 0.0002 --seed 1337
```

## Generation

Classic checkpoint:

```bash
python3 generate.py --ckpt runs/p2_refine/best.pt --prompt "Hello" --max_new_tokens 300
```

Modern checkpoint:

```bash
python3 generate_modern.py --ckpt runs/modern_moe_char_02/best.pt --prompt "Hello" --max_new_tokens 300
```

## What We Learned

- Under low data and 16GB memory constraints, training hygiene is as important as architecture.
- RMSNorm + RoPE + SwiGLU gave a major jump over the classic baseline.
- MoE gave a smaller but consistent improvement over dense modern controls.
- BPE was not automatically better in this tiny-corpus setup.
- Early stopping and stronger evaluation settings prevented wasted compute.

## Limitations

- Single training corpus (`data/input.txt`)
- Small model scale imposed by local memory limits
- Seed sensitivity on very small deltas
- No large external tokenizer corpus

## Recommended Next Step

Run 2-3 additional seeds on `modern_moe_char_02` to confirm whether `1.8729` is robust or seed noise.
