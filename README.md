# MiniGPT (Char-level) — Next-token Predictor in PyTorch (MPS friendly)

A small, readable decoder-only Transformer that learns **next-character prediction** end-to-end.



- End-to-end ML pipeline: **data → tokenization → batching → training → evaluation → generation**

```
input.txt → encode chars → batches (x, y=x shifted) → MiniGPT → logits → cross-entropy → AdamW update
                                         ↑
                                   causal mask
```

- Understanding of **logits / softmax / cross-entropy**, causal masking, and train vs eval behavior
- Practical training hygiene: gradient clipping, periodic evaluation, checkpointing
- **Hyperparameter tuning and regularization** to combat overfitting
- **Early stopping** with patience-based validation monitoring
- Simple CLI tooling (`train.py`, `generate.py`) and reproducible configuration

## Repo layout

- `notebooks/mini_gpt.ipynb` — learning notebook (step-by-step)
- `src/minigpt/`
  - `tokenizer.py` — char tokenizer (encode/decode + save/load)
  - `data.py` — dataset split + `get_batch`
  - `model.py` — MiniGPT + cached causal mask
  - `train.py` — training utilities (estimate loss, save checkpoints)
- `train.py` — CLI training entrypoint
- `generate.py` — CLI generation entrypoint
- `runs/` — experiment checkpoints and results

## Quickstart

### 1) Create a training text

Create a file at:

- `data/input.txt`

Any plain text works (book, articles, notes). Larger text = better generation.

### 2) Install deps

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3) Train

```bash
python train.py --input data/input.txt --max_steps 2000 --block_size 128
```

Checkpoints:

- `runs/<run_name>/best.pt` — best validation checkpoint (e.g. `runs/p2_refine/best.pt`)
- `runs/<run_name>/results.txt` — training metrics and config

### 4) Generate

```bash
python generate.py --ckpt runs/p2_refine/best.pt --prompt "Hello" --max_new_tokens 300
```

---

## Experiment Log — How I tuned to the best model

### Goal

Train a **char-level MiniGPT** next-token predictor and select the checkpoint that **generalizes best** using a validation split.

I split the text into **90% train / 10% val** (contiguous split) to measure generalization.

### What I observed early

- Training loss kept dropping, but **validation loss stopped improving and then got worse**.
- That's classic **overfitting** (memorizing the training text).
- So I focused on:
  - **Regularization** (dropout, weight decay)
  - **Reliable validation measurement** (more eval iters)
  - **Early stopping** (patience + min_delta)
  - Always saving `best.pt` when val improves

### Tuning steps (the path I followed)

1. **Baseline (scheduler + early stopping)**
   - Good learning, but val peaked around ~2400 then overfit hard.

2. **Add dropout (0.10 → 0.20)**
   - Big improvement: val kept improving longer and reached a lower minimum.

3. **Add weight decay (0.01 → 0.05)**
   - Improved generalization and reduced overfitting pressure.

4. **Make validation less noisy (`eval_iters` 50 → 80)**
   - More stable "best_val" selection.

5. **Allow smaller improvements (`min_delta` 0.001 → 0.0002)**
   - Prevented "missing" real but small improvements near the optimum.

6. **Test a slightly smaller LR (2e-4 → 1.5e-4)**
   - Slower training, but did **not** beat the best run.

**Final result:** `p2_refine` achieved the best validation loss.

**Best checkpoint:** `runs/p2_refine/best.pt` (best_val 1.9873 @ step 3200)

### Summary of runs

> **Metric:** `best_val` (lower is better) measured via `estimate_loss()` over `eval_iters` batches.
> **Checkpoint:** Each run saves `best.pt` at the best validation step.
> **stop_step:** Step where early stopping stopped training (not max_steps).

| Run                  | best_val ↓ | best_step | stop_step | block |  bs |     lr | layers | heads | embd |  dropout |       wd | eval_iters | patience |  min_delta | Notes                                                                       |
| -------------------- | ---------: | --------: | --------: | ----: | --: | -----: | -----: | ----: | ---: | -------: | -------: | ---------: | -------: | ---------: | --------------------------------------------------------------------------- |
| **p2_refine (BEST)** | **1.9873** |  **3200** |      7201 |   128 |  64 |   2e-4 |      4 |     4 |  128 | **0.20** | **0.05** |     **80** |       20 | **0.0002** | Best combo: stronger regularization + stable eval + fine-grained early stop |
| p2_do20_wd05         |     1.9876 |      3000 |      5000 |   128 |  64 |   2e-4 |      4 |     4 |  128 |     0.20 |     0.05 |         50 |       10 |      0.001 | Very close to best; slightly noisier eval + larger min_delta                |
| p1_base              |     2.0051 |      2400 |      4000 |   128 |  64 |   2e-4 |      4 |     4 |  128 |     0.10 |     0.01 |         50 |        8 |      0.001 | Baseline; overfits earlier                                                  |
| p2_lr1p5e4_refine    |     1.9884 |      3800 |      8800 |   128 |  64 | 1.5e-4 |      4 |     4 |  128 |     0.20 |     0.05 |         80 |       25 |     0.0002 | Slower LR didn't beat best (slightly worse minimum)                         |
| p3_lr1e4             |     2.0109 |      4200 |      6600 |   128 |  64 |   1e-4 |      4 |     4 |  128 |     0.10 |     0.01 |         50 |       12 |      0.001 | Lower LR helped stability but worse best_val than base                      |
| p4_med192            |     2.0884 |      1400 |      3400 |   128 |  32 |   2e-4 |      6 |     6 |  192 |     0.15 |     0.03 |         50 |       10 |      0.001 | Larger model overfit very fast for this dataset size                        |

### Best run details (p2_refine)

#### Final best metrics

- **best_val:** `1.9873`
- **best_step:** `3200`
- **final_train_loss (at stop):** `0.1443`
- **final_val_loss (at stop):** `3.0598`
- **total_steps (ran until early stop):** `7201`

> **Note:** `final_val_loss` is measured at the stop step. After best checkpoint (step 3200), training continues and the model overfits, so val rises. Use `best.pt` at `best_step` for inference.

#### Best configuration

| Parameter                 |            Value |
| ------------------------- | ---------------: |
| input_path                | `data/input.txt` |
| train_frac                |              0.9 |
| block_size                |              128 |
| n_layer / n_head / n_embd |      4 / 4 / 128 |
| dropout                   |         **0.20** |
| lr                        |         **2e-4** |
| weight_decay              |         **0.05** |
| grad_clip                 |              1.0 |
| batch_size                |               64 |
| eval_every                |              200 |
| eval_iters                |           **80** |
| patience                  |               20 |
| min_delta                 |       **0.0002** |
| seed                      |             1337 |

### Reproduce the best run

```bash
python train.py --input data/input.txt --out_dir runs/p2_refine \
  --block_size 128 --batch_size 64 --max_steps 50000 --lr 2e-4 \
  --n_layer 4 --n_head 4 --n_embd 128 --dropout 0.20 --weight_decay 0.05 \
  --eval_every 200 --eval_iters 80 --grad_clip 1.0 --seed 1337 \
  --patience 20 --min_delta 0.0002
```

---

## Sampling Settings (temperature & top_k)

The `generate.py` script uses **top-k sampling** with temperature control.

| Parameter       | Default | Range          | Effect                                                                                |
| --------------- | ------- | -------------- | ------------------------------------------------------------------------------------- |
| `--temperature` | 0.9     | 0.1 - 2.0      | Controls randomness: **lower = more deterministic**, **higher = more random/diverse** |
| `--top_k`       | 50      | 1 - vocab_size | Only sample from top-k most likely tokens                                             |

### Preset behaviors

| Setting                         | Effect                 | Use case                          |
| ------------------------------- | ---------------------- | --------------------------------- |
| `--temperature 0.5 --top_k 10`  | Conservative, coherent | Structured text, code-like output |
| `--temperature 0.9 --top_k 50`  | **(default)**          | Balanced general text generation  |
| `--temperature 1.2 --top_k 100` | Creative, diverse      | Creative writing, exploration     |

### Example outputs

Using the best checkpoint (`p2_refine/best.pt`):

**Conservative sampling (focused, coherent):**

```bash
python generate.py --ckpt runs/p2_refine/best.pt --prompt "The " \
  --max_new_tokens 200 --temperature 0.5 --top_k 10
```

Sample output:

```
The the Sillence was stems the was into be and the reat ournes.

And the pase the Silent a was not a star sigle feel even centuries that could not base the mole contines—not the mintals, sing the to Sun't
```

> **Note:** Char-level models often produce misspellings and artifacts. Coherence improves with more training data and larger context windows.

**Default sampling (balanced):**

```bash
python generate.py --ckpt runs/p2_refine/best.pt --prompt "Once upon" \
  --max_new_tokens 300 --temperature 0.9 --top_k 50
```

**Creative sampling (diverse):**

```bash
python generate.py --ckpt runs/p2_refine/best.pt --prompt "Hello " \
  --max_new_tokens 400 --temperature 1.2 --top_k 100
```

---


## Notes

- Char-level tokenization is chosen for clarity (not speed).
- Production LLMs use subword tokenizers (BPE), but the training loop is the same idea.
- **Overfitting is expected** with small datasets — focus on the best validation checkpoint, not final weights.

---

## Recommended experiments

These are great for learning:

1. **Overfit test:** Train on a tiny subset (first 2k characters) until loss → near zero.
2. **Ablation:** Change `block_size` (64 vs 256) and report val loss differences.
3. **Capacity:** Test `n_layer` 2 vs 6 and compare quality vs speed.
4. **Sampling sweep:** Generate with temperature 0.3, 0.7, 1.5 and show the difference.
5. **Train vs eval:** Demonstrate dropout effect (predictions vary in train mode, stable in eval).

