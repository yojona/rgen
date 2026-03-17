#!/usr/bin/env python3
"""
Tokenize raw JSONL datasets into binary .bin files (section 13.2 of spec).

Output format: flat array of uint16 tokens.  Documents are separated by the
tokenizer's EOS token.  Files can be memory-mapped with np.memmap during
training so the full dataset never needs to load into RAM.

Usage:
  python data/prepare.py --phase 1        # tokenize phase 1 datasets
  python data/prepare.py --phase 2        # tokenize phase 2 datasets
  python data/prepare.py --phase all      # both
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from tqdm import tqdm

# LLaMA tokens fit in uint16 (max 65535) — saves 50% vs int32
DTYPE = np.uint16

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR / "raw"
TOKENIZED_DIR = SCRIPT_DIR / "tokenized"
TOKENIZER_DIR = SCRIPT_DIR / "tokenizer"


# ---------------------------------------------------------------------------
# Reasoning filter (applied to phase 1 data only)
# ---------------------------------------------------------------------------

_REASONING_INDICATORS = [
    "por lo tanto", "entonces", "dado que", "porque",
    "se concluye", "si", "paso 1", "paso 2",
    "therefore", "thus", "since", "because",
    "step 1", "step 2", "first", "then",
]


def filter_reasoning(text: str) -> bool:
    """Return True if *text* contains at least 2 reasoning indicators."""
    text_lower = text.lower()
    return sum(1 for ind in _REASONING_INDICATORS if ind in text_lower) >= 2


# ---------------------------------------------------------------------------
# Instruction formatting (phase 2)
# ---------------------------------------------------------------------------

def format_instruction(example: dict) -> str:
    """Convert a phase-2 example to the unified instruction format."""
    pregunta = example.get("pregunta", example.get("problem", ""))
    respuesta = example.get("respuesta", example.get("solution", ""))
    sistema = example.get("sistema", "Eres un asistente que razona paso a paso.")

    return (
        f"<|system|>\n{sistema}\n"
        f"<|user|>\n{pregunta}\n"
        f"<|assistant|>\n{respuesta}\n"
        f"<|end|>"
    )


# ---------------------------------------------------------------------------
# Core tokenization
# ---------------------------------------------------------------------------

def _count_lines(path: Path) -> int:
    """Fast line count for progress bar."""
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return count


def tokenize_file(
    input_path: Path,
    output_path: Path,
    tokenizer,
    filter_fn: Optional[Callable[[str], bool]] = None,
    text_key: str = "text",
    max_docs: Optional[int] = None,
) -> int:
    """Tokenize a JSONL file and write tokens as a flat uint16 binary.

    Each document is terminated with the tokenizer's EOS token so the
    trainer can tell documents apart.  The binary file is written in
    chunks to avoid accumulating millions of tokens in RAM.

    Args:
        input_path:  JSONL source file.
        output_path: .bin destination file.
        tokenizer:   HuggingFace tokenizer (needs .encode and .eos_token_id).
        filter_fn:   optional predicate — text is skipped when it returns False.
        text_key:    JSON key that holds the text payload.
        max_docs:    stop after this many documents (None = all).

    Returns:
        Total number of tokens written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_lines = _count_lines(input_path)

    token_count = 0
    doc_count = 0
    filtered_count = 0
    CHUNK_SIZE = 50_000  # flush every N tokens

    buffer: list[int] = []

    with open(output_path, "wb") as out_f, \
         open(input_path, "r") as in_f:
        for line in tqdm(in_f, total=total_lines, desc=input_path.name):
            if max_docs is not None and doc_count >= max_docs:
                break

            try:
                example = json.loads(line)
                text = example[text_key]
            except (json.JSONDecodeError, KeyError):
                continue

            if filter_fn is not None and not filter_fn(text):
                filtered_count += 1
                continue

            tokens = tokenizer.encode(text, add_special_tokens=False)
            tokens.append(tokenizer.eos_token_id)

            buffer.extend(tokens)
            doc_count += 1
            token_count += len(tokens)

            if len(buffer) >= CHUNK_SIZE:
                arr = np.array(buffer, dtype=DTYPE)
                out_f.write(arr.tobytes())
                buffer.clear()

        # Final flush
        if buffer:
            arr = np.array(buffer, dtype=DTYPE)
            out_f.write(arr.tobytes())

    size_mb = output_path.stat().st_size / 1e6
    print(f"  Documents processed: {doc_count:,}")
    if filtered_count:
        print(f"  Documents filtered:  {filtered_count:,}")
    print(f"  Tokens total:        {token_count:,} ({token_count / 1e9:.3f}B)")
    print(f"  Size on disk:        {size_mb:.1f} MB")
    return token_count


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def prepare_phase1(tokenizer) -> None:
    print("\n=== PHASE 1: Pretraining ===\n")

    # 1. TinyStories — no filter, already clean text
    ts_path = RAW_DIR / "tinystories.jsonl"
    if ts_path.exists():
        print("[1/3] TinyStories")
        tokenize_file(ts_path, TOKENIZED_DIR / "tinystories.bin", tokenizer)
    else:
        print(f"[1/3] Skipping TinyStories (not found: {ts_path})")

    # 2. Wikipedia ES — reasoning filter
    wiki_path = RAW_DIR / "wikipedia_es.jsonl"
    if wiki_path.exists():
        print("\n[2/3] Wikipedia ES (with reasoning filter)")
        tokenize_file(
            wiki_path, TOKENIZED_DIR / "wikipedia_es.bin", tokenizer,
            filter_fn=filter_reasoning,
        )
    else:
        print(f"[2/3] Skipping Wikipedia ES (not found: {wiki_path})")

    # 3. OpenWebMath — reasoning filter
    owm_path = RAW_DIR / "openwebmath.jsonl"
    if owm_path.exists():
        print("\n[3/3] OpenWebMath (with reasoning filter)")
        tokenize_file(
            owm_path, TOKENIZED_DIR / "openwebmath.bin", tokenizer,
            filter_fn=filter_reasoning,
        )
    else:
        print(f"[3/3] Skipping OpenWebMath (not found: {owm_path})")


def prepare_phase2(tokenizer) -> None:
    print("\n=== PHASE 2: Reasoning fine-tuning ===\n")
    phase2_raw = RAW_DIR / "phase2"
    phase2_tok = TOKENIZED_DIR / "phase2"

    for name in ("numinamath", "logiqa", "openhermes_raw"):
        input_path = phase2_raw / f"{name}.jsonl"
        if not input_path.exists():
            print(f"Skipping {name} (not found: {input_path})")
            continue

        output_path = phase2_tok / f"{name}.bin"
        print(f"[phase2] {name}")

        # Reformat each example into unified instruction format, write to
        # a temporary JSONL, then tokenize that.
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jsonl")
        try:
            with os.fdopen(tmp_fd, "w") as tmp_f, open(input_path) as src_f:
                for line in src_f:
                    try:
                        example = json.loads(line)
                        formatted = format_instruction(example)
                        tmp_f.write(json.dumps({"text": formatted}) + "\n")
                    except (json.JSONDecodeError, KeyError):
                        continue
            tokenize_file(Path(tmp_path), output_path, tokenizer)
        finally:
            os.unlink(tmp_path)
        print()

    # Synthetic data (if generated)
    syn_path = phase2_raw / "synthetic_es.jsonl"
    if syn_path.exists():
        print("[phase2] synthetic_es")
        tokenize_file(syn_path, phase2_tok / "synthetic_es.bin", tokenizer)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tokenize raw JSONL datasets into binary .bin files.",
    )
    parser.add_argument(
        "--phase",
        choices=["1", "2", "all"],
        default="all",
        help="Which phase to tokenize (default: all).",
    )
    args = parser.parse_args()

    print("Loading tokenizer ...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    print(f"Tokenizer ready: vocab_size={tokenizer.vocab_size}\n")

    if args.phase in ("1", "all"):
        prepare_phase1(tokenizer)
    if args.phase in ("2", "all"):
        prepare_phase2(tokenizer)

    print("\nDone. Tokenized files in", TOKENIZED_DIR)


if __name__ == "__main__":
    main()
