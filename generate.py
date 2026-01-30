from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path so we can import minigpt
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch

from minigpt.data import select_device
from minigpt.model import GPTConfig, MiniGPT
from minigpt.tokenizer import CharTokenizer


def main():
    p = argparse.ArgumentParser(description="Generate text from a trained MiniGPT checkpoint.")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt).")
    p.add_argument("--prompt", default="Hello", help="Prompt text.")
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=50)
    args = p.parse_args()

    device = select_device(prefer_mps=True)
    ckpt = torch.load(args.ckpt, map_location=device)

    tok = CharTokenizer(chars=ckpt["tokenizer"]["chars"])
    gpt_cfg = GPTConfig(**ckpt["gpt_cfg"])
    model = MiniGPT(gpt_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    idx = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
