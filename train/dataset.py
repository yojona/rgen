"""
Memory-mapped dataset for RGEN pretraining.

Loads tokenized .bin files (uint16 arrays) via np.memmap so the full
dataset never needs to fit in RAM.  Returns sliding windows of
max_seq_len tokens as (input, target) pairs for next-token prediction.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset


class MemmapTokenDataset(Dataset):
    """PyTorch Dataset over a single memory-mapped .bin token file.

    Each sample is a contiguous window of (max_seq_len + 1) tokens.
    The first max_seq_len tokens are the input; the last max_seq_len
    tokens (shifted by 1) are the target.

    The windows are non-overlapping by default to avoid seeing the
    same tokens twice per epoch.  Set stride < max_seq_len for
    overlapping windows if desired.
    """

    def __init__(
        self,
        path: Union[str, Path],
        max_seq_len: int,
        stride: int = 0,
    ):
        self.path = Path(path)
        self.max_seq_len = max_seq_len
        self.stride = stride if stride > 0 else max_seq_len

        # Memory-map the file (read-only, never loads into RAM)
        self.data = np.memmap(self.path, dtype=np.uint16, mode="r")
        self.n_tokens = len(self.data)

        # Number of non-overlapping windows we can extract
        # Each window needs (max_seq_len + 1) tokens for input/target shift
        window = self.max_seq_len + 1
        if self.n_tokens < window:
            self.n_samples = 0
        else:
            self.n_samples = (self.n_tokens - window) // self.stride + 1

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        end = start + self.max_seq_len + 1
        chunk = self.data[start:end].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])   # input:  [max_seq_len]
        y = torch.from_numpy(chunk[1:])    # target: [max_seq_len]
        return x, y

    def __repr__(self) -> str:
        return (
            f"MemmapTokenDataset({self.path.name}, "
            f"tokens={self.n_tokens:,}, samples={self.n_samples:,}, "
            f"seq_len={self.max_seq_len})"
        )


def build_dataset(
    paths: List[Union[str, Path]],
    max_seq_len: int,
    stride: int = 0,
) -> ConcatDataset:
    """Build a ConcatDataset from multiple .bin files."""
    datasets = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"  [warn] skipping {p} (not found)")
            continue
        ds = MemmapTokenDataset(p, max_seq_len, stride)
        if len(ds) == 0:
            print(f"  [warn] skipping {p} (too small: {ds.n_tokens:,} tokens)")
            continue
        datasets.append(ds)
        print(f"  {ds}")
    return ConcatDataset(datasets)
