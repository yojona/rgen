"""Tokenizer wrapper around sentencepiece / HuggingFace tokenizer.

Loads the LLaMA vocabulary from data/tokenizer/ (downloaded via
`python data/download.py --tokenizer-only`).  Falls back to
sentencepiece directly if a .model file is provided.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

_DEFAULT_TOKENIZER_DIR = Path(__file__).resolve().parent.parent / "data" / "tokenizer"


class Tokenizer:
    """Unified tokenizer that wraps either a HuggingFace tokenizer dir
    or a raw sentencepiece .model file."""

    def __init__(self, path: Optional[str | Path] = None):
        """
        Args:
            path: one of:
              - directory containing a HuggingFace tokenizer (tokenizer.json / tokenizer.model)
              - path to a .model sentencepiece file
              - None → uses default at data/tokenizer/
        """
        path = Path(path) if path is not None else _DEFAULT_TOKENIZER_DIR

        if path.is_dir():
            self._init_from_hf_dir(path)
        elif path.suffix == ".model" and path.is_file():
            self._init_from_sp(path)
        else:
            raise FileNotFoundError(
                f"Tokenizer not found at {path}. Run:\n"
                "  python data/download.py --tokenizer-only"
            )

    # ----- HuggingFace backend -----

    def _init_from_hf_dir(self, dir_path: Path) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("Install transformers: pip install transformers")

        self._backend = "hf"
        self._tok = AutoTokenizer.from_pretrained(str(dir_path))

        self.vocab_size: int = self._tok.vocab_size
        self.bos_id: int = self._tok.bos_token_id or 1
        self.eos_id: int = self._tok.eos_token_id or 2
        self.pad_id: int = self._tok.pad_token_id if self._tok.pad_token_id is not None else self.eos_id

    # ----- SentencePiece backend -----

    def _init_from_sp(self, model_path: Path) -> None:
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError("Install sentencepiece: pip install sentencepiece")

        self._backend = "sp"
        self._tok = spm.SentencePieceProcessor()
        self._tok.Load(str(model_path))

        self.vocab_size: int = self._tok.GetPieceSize()
        self.bos_id: int = self._tok.bos_id()
        self.eos_id: int = self._tok.eos_id()
        self.pad_id: int = self.eos_id  # sentencepiece has no pad; reuse eos

    # ----- Public API -----

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode text to token ids."""
        if self._backend == "hf":
            ids = self._tok.encode(text, add_special_tokens=False)
        else:
            ids = self._tok.Encode(text)

        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode token ids back to text."""
        if self._backend == "hf":
            return self._tok.decode(ids, skip_special_tokens=skip_special)
        else:
            return self._tok.Decode(ids)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"Tokenizer(backend={self._backend!r}, vocab_size={self.vocab_size})"
