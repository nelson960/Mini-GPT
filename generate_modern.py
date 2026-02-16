from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch

from minigpt.data import select_device
from minigpt.modern_model import ModernGPT, ModernGPTConfig
from minigpt.modern_tokenizer import tokenizer_from_payload


def main() -> None:
    p = argparse.ArgumentParser(description="Generate text from a trained modern MiniGPT checkpoint.")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt)")
    p.add_argument("--prompt", default="Hello", help="Prompt text")
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=50)
    args = p.parse_args()

    device = select_device(prefer_mps=True)
    ckpt = torch.load(args.ckpt, map_location=device)

    tok = tokenizer_from_payload(ckpt["tokenizer"])
    cfg_key = "model_cfg" if "model_cfg" in ckpt else "gpt_cfg"
    model_cfg = ModernGPTConfig(**ckpt[cfg_key])
    model = ModernGPT(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    idx = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
