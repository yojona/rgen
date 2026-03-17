#!/usr/bin/env python3
"""Interactive multi-turn chat with a trained RGEN model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from rgen.config import RGENConfig
from rgen.model import RGEN
from rgen.tokenizer import Tokenizer


SPECIAL_TOKENS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "end": "<|end|>",
}


def build_prompt(
    history: list[dict[str, str]],
    system_prompt: str | None = None,
) -> str:
    """Build the full prompt string from conversation history.

    Args:
        history: list of {"role": "user"|"assistant", "content": "..."}
        system_prompt: optional system message prepended once
    """
    parts = []
    if system_prompt:
        parts.append(f"{SPECIAL_TOKENS['system']}\n{system_prompt}")
    for turn in history:
        tag = SPECIAL_TOKENS[turn["role"]]
        parts.append(f"{tag}\n{turn['content']}")
    # Prompt the model to start generating as assistant
    parts.append(SPECIAL_TOKENS["assistant"])
    return "\n".join(parts)


def generate_reply(
    model: RGEN,
    tokenizer: Tokenizer,
    prompt_text: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    """Run generation and extract the assistant reply."""
    ids = tokenizer.encode(prompt_text, add_bos=True)
    input_ids = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    full_text = tokenizer.decode(output_ids[0].tolist())

    # Extract only the new assistant response
    # The model should stop at <|end|> or <|user|>
    reply = full_text[len(prompt_text) :]
    for stop in (SPECIAL_TOKENS["end"], SPECIAL_TOKENS["user"]):
        if stop in reply:
            reply = reply[: reply.index(stop)]
    return reply.strip()


def main():
    parser = argparse.ArgumentParser(description="Chat with RGEN")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt checkpoint")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max tokens per reply")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--system", type=str, default=None,
                        help="System prompt (optional)")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer")
    args = parser.parse_args()

    # Load model
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mcfg = RGENConfig(**ckpt["model_config"])
    model = RGEN(mcfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    params = mcfg.estimated_params()["total_M"]
    print(f"RGEN {params}M params | step {ckpt['step']} | temp {args.temperature}")
    print("Type 'quit' to exit, 'reset' to clear history.\n")

    tok = Tokenizer(args.tokenizer)
    history: list[dict[str, str]] = []

    while True:
        try:
            user_input = input("tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "reset":
            history.clear()
            print("(historial limpio)\n")
            continue

        history.append({"role": "user", "content": user_input})

        prompt_text = build_prompt(history, system_prompt=args.system)
        reply = generate_reply(
            model, tok, prompt_text,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

        history.append({"role": "assistant", "content": reply})
        print(f"rgen: {reply}\n")


if __name__ == "__main__":
    main()
