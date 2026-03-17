#!/usr/bin/env python3
"""Interactive text generation with a trained RGEN model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from rgen.config import RGENConfig
from rgen.model import RGEN
from rgen.tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate text with RGEN")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt checkpoint")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="Text prompt")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer")
    args = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mcfg = RGENConfig(**ckpt["model_config"])
    model = RGEN(mcfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded model: {mcfg.estimated_params()['total_M']}M params, step {ckpt['step']}")

    # Load tokenizer
    tok = Tokenizer(args.tokenizer)

    # Encode prompt
    ids = tok.encode(args.prompt, add_bos=True)
    input_ids = torch.tensor([ids], dtype=torch.long)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    text = tok.decode(output_ids[0].tolist())
    print(f"\nPrompt: {args.prompt}")
    print(f"Generated:\n{text}")


if __name__ == "__main__":
    main()
