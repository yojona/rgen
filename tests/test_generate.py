"""Validate autoregressive generation works correctly."""

import torch
from rgen.config import TINY_CONFIG
from rgen.model import RGEN


def test_generates_n_tokens():
    """Model should generate exactly max_new_tokens without errors."""
    config = TINY_CONFIG
    model = RGEN(config)
    model.eval()

    batch, prompt_len = 1, 8
    max_new = 16
    input_ids = torch.randint(0, config.vocab_size, (batch, prompt_len))

    output = model.generate(input_ids, max_new_tokens=max_new, temperature=1.0)

    expected_len = prompt_len + max_new
    assert output.shape == (batch, expected_len), \
        f"Expected shape {(batch, expected_len)}, got {output.shape}"
    assert (output >= 0).all() and (output < config.vocab_size).all(), \
        "Generated token ids out of vocab range"
    print(f"✓ Generated {max_new} tokens: output shape {output.shape}")


def test_greedy_is_deterministic():
    """Greedy decoding (temperature=0) must produce identical results."""
    config = TINY_CONFIG
    model = RGEN(config)
    model.eval()

    batch, prompt_len = 1, 8
    input_ids = torch.randint(0, config.vocab_size, (batch, prompt_len))

    out1 = model.generate(input_ids, max_new_tokens=10, temperature=0)
    out2 = model.generate(input_ids, max_new_tokens=10, temperature=0)

    assert torch.equal(out1, out2), "Greedy decoding is not deterministic"
    print(f"✓ Greedy decoding is deterministic")


def test_temperature_sampling():
    """With temperature > 0, different seeds should (usually) give different outputs."""
    config = TINY_CONFIG
    model = RGEN(config)
    model.eval()

    batch, prompt_len = 1, 8
    input_ids = torch.randint(0, config.vocab_size, (batch, prompt_len))

    # Run multiple times with high temperature — at least one should differ
    outputs = []
    for seed in range(5):
        torch.manual_seed(seed)
        out = model.generate(input_ids, max_new_tokens=10, temperature=2.0)
        outputs.append(out)

    all_same = all(torch.equal(outputs[0], o) for o in outputs[1:])
    assert not all_same, "All 5 samples are identical at temperature=2.0 — sampling may be broken"
    print(f"✓ Temperature sampling produces varied outputs")


def test_batched_generation():
    """Generation should work with batch_size > 1."""
    config = TINY_CONFIG
    model = RGEN(config)
    model.eval()

    batch, prompt_len = 3, 8
    max_new = 10
    input_ids = torch.randint(0, config.vocab_size, (batch, prompt_len))

    output = model.generate(input_ids, max_new_tokens=max_new, temperature=0)

    assert output.shape == (batch, prompt_len + max_new), \
        f"Expected shape {(batch, prompt_len + max_new)}, got {output.shape}"
    print(f"✓ Batched generation works: {output.shape}")


def test_top_k_sampling():
    """Top-k should restrict sampling to k most likely tokens."""
    config = TINY_CONFIG
    model = RGEN(config)
    model.eval()

    batch, prompt_len = 1, 8
    input_ids = torch.randint(0, config.vocab_size, (batch, prompt_len))

    torch.manual_seed(42)
    output = model.generate(input_ids, max_new_tokens=10, temperature=1.0, top_k=5)

    assert output.shape[1] == prompt_len + 10
    print(f"✓ Top-k sampling runs without errors")


if __name__ == "__main__":
    test_generates_n_tokens()
    test_greedy_is_deterministic()
    test_temperature_sampling()
    test_batched_generation()
    test_top_k_sampling()
    print("\nAll generation tests passed!")
