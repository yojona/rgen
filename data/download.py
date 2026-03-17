#!/usr/bin/env python3
"""
Download datasets for RGEN training (section 13.1 of spec).

Phase 1 — Pretraining (linguistic competence):
  - TinyStories       (~476M tokens, validates pipeline fast)
  - Wikipedia ES       (~200M tokens, Spanish base)
  - OpenWebMath        (~1B tokens, streaming, math reasoning)

Phase 2 — Reasoning fine-tuning:
  - NuminaMath         (~860K examples, step-by-step math)
  - LogiQA             (~8K examples, formal logic)
  - OpenHermes 2.5     (~500K examples, general reasoning)

Usage:
  python data/download.py --tokenizer-only     # just the LLaMA tokenizer
  python data/download.py --phase 1            # phase 1 datasets
  python data/download.py --phase 2            # phase 2 datasets
  python data/download.py --phase all          # everything
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Resolve paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR / "raw"
TOKENIZER_DIR = SCRIPT_DIR / "tokenizer"


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def download_tokenizer() -> None:
    """Download the LLaMA tokenizer (vocabulary only, not the model weights).

    Requires: pip install transformers sentencepiece protobuf
    Note: LLaMA tokenizer on HuggingFace may require authentication.
    If meta-llama/Llama-2-7b-hf is gated, we fall back to an open
    sentencepiece-compatible tokenizer with the same 32K vocab.
    """
    from transformers import AutoTokenizer

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    # Try the official LLaMA tokenizer first; fall back to an ungated alternative
    candidates = [
        "meta-llama/Llama-2-7b-hf",
        "NousResearch/Llama-2-7b-hf",  # community mirror, no gate
    ]

    for name in candidates:
        try:
            print(f"Trying tokenizer: {name}")
            tok = AutoTokenizer.from_pretrained(name)
            tok.save_pretrained(str(TOKENIZER_DIR))
            print(f"Tokenizer saved to {TOKENIZER_DIR}  (vocab_size={tok.vocab_size})")
            return
        except Exception as e:
            print(f"  skipped ({e})")

    print(
        "ERROR: Could not download any LLaMA tokenizer.\n"
        "Options:\n"
        "  1. Accept the LLaMA license at https://huggingface.co/meta-llama/Llama-2-7b-hf\n"
        "     then run: huggingface-cli login\n"
        "  2. Place a sentencepiece .model file in data/tokenizer/ manually.",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Phase 1 datasets
# ---------------------------------------------------------------------------

def download_tiny_stories(output_path: Path) -> None:
    """Download TinyStories — small, clean, fast to validate pipeline.

    ~2.1M short children's stories.
    Time: ~2 min.  Disk: ~500 MB.
    """
    from datasets import load_dataset

    print("Downloading TinyStories ...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for example in dataset:
            f.write(json.dumps({"text": example["text"]}) + "\n")

    print(f"TinyStories done: {len(dataset):,} stories -> {output_path}")


def download_wikipedia_es(output_path: Path, min_length: int = 2000) -> None:
    """Download Spanish Wikipedia, filtering out short articles.

    Articles < min_length chars are mostly lists/tables — not useful for prose.
    Time: ~8 min.  Disk: ~1.2 GB.
    """
    from datasets import load_dataset

    print("Downloading Wikipedia ES ...")
    dataset = load_dataset("wikimedia/wikipedia", "20231101.es", split="train")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped = 0
    with open(output_path, "w") as f:
        for example in dataset:
            if len(example["text"]) >= min_length:
                f.write(json.dumps({"text": example["text"]}) + "\n")
                kept += 1
            else:
                skipped += 1

    print(f"Wikipedia ES done: {kept:,} articles kept, {skipped:,} short ones skipped -> {output_path}")


def download_openwebmath(output_path: Path, max_chars: int = 1_000_000_000) -> None:
    """Download OpenWebMath via streaming, stopping at ~1B characters.

    Uses streaming=True so it never tries to load the full 14B-token
    dataset into memory.
    Time: ~45 min.  Disk: ~4 GB.
    """
    from datasets import load_dataset

    print("Downloading OpenWebMath (streaming) ...")
    dataset = load_dataset(
        "open-web-math/open-web-math",
        split="train",
        streaming=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    char_count = 0
    doc_count = 0
    with open(output_path, "w") as f:
        for example in dataset:
            text = example["text"]
            f.write(json.dumps({"text": text}) + "\n")
            char_count += len(text)
            doc_count += 1
            if doc_count % 10_000 == 0:
                print(f"  {doc_count:,} docs, ~{char_count / 1e9:.2f}B chars")
            if char_count >= max_chars:
                break

    print(f"OpenWebMath done: {doc_count:,} docs -> {output_path}")


# ---------------------------------------------------------------------------
# Phase 2 datasets
# ---------------------------------------------------------------------------

def download_phase2_datasets(output_dir: Path) -> None:
    """Download all Phase 2 reasoning fine-tuning datasets."""
    from datasets import load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- NuminaMath: step-by-step math solutions ---
    print("Downloading NuminaMath ...")
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    out = output_dir / "numinamath.jsonl"
    with open(out, "w") as f:
        for ex in ds:
            f.write(json.dumps({
                "pregunta": ex["problem"],
                "respuesta": ex["solution"],
            }) + "\n")
    print(f"  NuminaMath done: {len(ds):,} examples -> {out}")

    # --- LogiQA: formal logic in natural language ---
    print("Downloading LogiQA ...")
    ds = load_dataset("lucasmccabe/logiqa", split="train")
    out = output_dir / "logiqa.jsonl"
    with open(out, "w") as f:
        for ex in ds:
            f.write(json.dumps({
                "contexto": ex["context"],
                "pregunta": ex["query"],
                "opciones": ex["options"],
                "respuesta": ex["correct_option"],
            }) + "\n")
    print(f"  LogiQA done: {len(ds):,} examples -> {out}")

    # --- OpenHermes 2.5: general reasoning (to be filtered later) ---
    print("Downloading OpenHermes 2.5 ...")
    ds = load_dataset("teknium/OpenHermes-2.5", split="train")
    out = output_dir / "openhermes_raw.jsonl"
    with open(out, "w") as f:
        for ex in ds:
            f.write(json.dumps({
                "sistema": ex.get("system_prompt", ""),
                "conversacion": ex["conversations"],
            }) + "\n")
    print(f"  OpenHermes done: {len(ds):,} examples (unfiltered) -> {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download datasets for RGEN training.",
    )
    parser.add_argument(
        "--phase",
        choices=["1", "2", "all"],
        default="all",
        help="Which training phase datasets to download (default: all).",
    )
    parser.add_argument(
        "--tokenizer-only",
        action="store_true",
        help="Download only the LLaMA tokenizer, skip datasets.",
    )
    args = parser.parse_args()

    if args.tokenizer_only:
        download_tokenizer()
        return

    if args.phase in ("1", "all"):
        download_tiny_stories(RAW_DIR / "tinystories.jsonl")
        download_wikipedia_es(RAW_DIR / "wikipedia_es.jsonl")
        download_openwebmath(RAW_DIR / "openwebmath.jsonl")

    if args.phase in ("2", "all"):
        download_phase2_datasets(RAW_DIR / "phase2")


if __name__ == "__main__":
    main()
