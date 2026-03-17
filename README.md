# RGEN: Recursive Generative Network

> Tiny bilingual model that thinks in silence before speaking, with zero extra context tokens.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-orange.svg)](https://pytorch.org)

---

## What is RGEN?

RGEN is a small language model with a novel recursive reasoning architecture. Instead of reasoning through visible chain-of-thought tokens (like o1 or DeepSeek-R1), RGEN reasons **silently**, a dedicated Reasoner module iterates K times over the input, producing a latent reasoning state `z*` that conditions the Generator through cross-attention.

```
Standard CoT model:              RGEN:
  User: What is 17 × 23?           User: What is 17 × 23?
  Model: <thinking>                Model: [Reasoner runs K=8 times internally]
    17 × 20 = 340                  The answer is 391.
    17 × 3 = 51
    340 + 51 = 391
  </thinking>
  The answer is 391.

  → 80 tokens consumed             → 6 tokens consumed
```

The reasoning happens in `z*`, a continuous vector that never becomes visible text. Zero extra context tokens consumed per response.

---

## Architecture
<img width="975" height="726" alt="image" src="https://github.com/user-attachments/assets/703314df-f614-4097-9f50-f4fea698c221" />

```
INPUT TEXT
    │
    ▼
[Embedding + RoPE]
    │
    ├─────────────────────────────────────────┐
    │                                         │
    ▼                                         ▼
┌─────────────────────┐         ┌─────────────────────────┐
│     REASONER        │         │      GENERATOR          │
│                     │         │                         │
│  z₀ = zeros         │         │  Autoregressive         │
│  for k in range(K): │         │  Transformer Decoder    │
│    z_k = f(x, z_k-1)│─ z* ──▶ │                         │
│                     │         │  Self-attention         │
│  Output: z*         │         │  Cross-attention(z*)    │
│  [2 layers × K=8]   │         │  FFN                    │
└─────────────────────┘         └─────────────────────────┘
                                          │
                                          ▼
                                    OUTPUT TOKENS
```

**Key insight:** The Reasoner uses **weight sharing** — 2 layers applied K=8 times. This gives an effective depth of 16 reasoning steps with the parameters of only 2 layers.

---

## Two Configs

| | RGEN-Small | RGEN-Nano |
|---|---|---|
| Parameters | ~70M | ~8M |
| Size (4-bit) | ~35MB | ~4MB |
| Vocabulary | 32K (LLaMA) | 16K (own) |
| d_model | 512 | 192 |
| Generator layers | 6 | 3 |
| Knowledge | Medium | None (tools) |
| Tool use | Phase 4 | Native (Phase 3) |
| Training tokens | ~6.5B | ~600M |
| Cloud cost (H100) | ~$38 USD | ~$21 USD |
| Languages | ES + EN | ES + EN |

**RGEN-Small:** broader language base, better for standalone use.
**RGEN-Nano:** designed to know nothing — knowledge comes through web search and tools. Smaller, faster, embeddable anywhere.

---

## Why does this matter?

Current approaches to reasoning in small models:

| Approach | Method | Cost |
|---|---|---|
| Chain-of-thought | Reasoning as visible text | Burns context window |
| o1 / R1 style | "Thinking tokens" | Extra tokens per response |
| TRM / HRM (2025) | Recursive reasoning | Works only on discrete puzzles |
| **RGEN** | **Latent recursive reasoning** | **Zero extra tokens, generative** |

TRM and HRM ([arXiv:2510.04871](https://arxiv.org/abs/2510.04871), [arXiv:2506.21734](https://arxiv.org/abs/2506.21734)) demonstrated that recursive reasoning works at 7-27M parameters — but only for discrete puzzles (Sudoku, mazes). RGEN extends recursive reasoning to open-ended text generation for the first time.

---

## Status

- [x] Architecture implemented and validated
- [x] Training pipeline working on Apple Silicon (MPS)
- [x] Three training phases verified (generator_only → reasoner_warmup → joint)
- [x] Reasoner activates correctly — z* diversity confirmed (negative div loss)
- [x] Text generation working after Phase 1 validation
- [ ] Full Phase 1 training (cloud)
- [ ] Phase 2 reasoning fine-tuning
- [ ] Phase 3 conversational tuning
- [ ] Benchmarks (GSM8K, LogiQA, ARC)
- [ ] MLX 4-bit inference on Apple Silicon
- [ ] RGEN-Nano with native tool use

---

## Training Phases

```
Phase 1 — Language base
  Datasets: TinyStories + Wikipedia ES + OpenWebMath (Small)
            TinyStories + OpenSubtitles ES + DailyDialog (Nano)
  Goal: learn grammar and coherence

Phase 2 — Reasoning fine-tuning
  Datasets: NuminaMath + LogiQA + Synthetic CoT (Claude-distilled)
  Goal: structured step-by-step reasoning

Phase 3 — Conversational tuning
  Datasets: OpenSubtitles + DailyDialog + Synthetic conversations
  Goal: natural register, "I don't know" when appropriate

Phase 4 (Small only) — Tool use
  Datasets: Synthetic tool-use examples
  Goal: web search, calculator integration
```

---

## Quick Start

```bash
git clone https://github.com/yojona/rgen.git
cd rgen
pip install -r requirements.txt

# Download tokenizer
python3 data/download.py --tokenizer-only

# Download and prepare Phase 1 data
python3 data/download.py --phase 1
python3 data/prepare.py --phase 1

# Validate architecture (should show 0 NaN in 119 activations)
python3 tests/test_shapes.py
python3 tests/test_reasoner.py

# Smoke test training (10 steps)
python3 train/train.py --config config/small.yaml --max-steps 10

# Generate text (after training)
python3 eval/generate.py \
  --checkpoint checkpoints/checkpoint_best.pt \
  --prompt "Había una vez" \
  --max-tokens 100
```

---

## Cloud Training (~$38 USD for Small, ~$21 USD for Nano)

```bash
# On Lambda Labs H100 instance
git clone https://github.com/yojona/rgen.git
cd rgen
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Phase 1 (~6 hours)
python3 data/download.py --phase 1
python3 data/prepare.py --phase 1
nohup python3 train/train.py \
  --config config/small.yaml \
  --output-dir checkpoints/phase1 \
  > logs/phase1.log 2>&1 &
```
---

## Benchmarks (target)

| Benchmark | RGEN-Small target | RGEN-Nano target | GPT-2 small ref |
|---|---|---|---|
| LogiQA | >35% | >30% | ~25% |
| GSM8K | >5% | >3% | ~0% |
| ARC-Easy | >50% | >45% | ~45% |

Results will be updated after full training completes.
---

## Citation

If you use RGEN in your research:

```bibtex
@misc{rgen2026,
  title={RGEN: Recursive Generative Network with Latent Reasoning},
  author={Jonathan Ayala},
  year={2026},
  note={Work in progress. https://github.com/yojona/rgen}
}
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
