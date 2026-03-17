"""Validate that the Reasoner's recursive loop works correctly."""

import torch
from rgen.config import TINY_CONFIG
from rgen.model import RGEN


def test_z_changes_each_iteration():
    """z_k must differ from z_{k-1} — the loop must actually do work."""
    config = TINY_CONFIG
    model = RGEN(config)
    model.eval()

    batch, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))

    with torch.no_grad():
        x = model.embedding(input_ids)
        _, intermediates = model.reasoner(x, return_intermediates=True)

    assert len(intermediates) == config.reasoner_iterations, \
        f"Expected {config.reasoner_iterations} intermediates, got {len(intermediates)}"

    # Each z_k should differ from z_{k-1}
    for k in range(1, len(intermediates)):
        diff = (intermediates[k] - intermediates[k - 1]).abs().max().item()
        assert diff > 1e-6, f"z_{k} is identical to z_{k-1} (max diff={diff})"

    # z_0 should differ from zeros (the initial state)
    z0_magnitude = intermediates[0].abs().max().item()
    assert z0_magnitude > 1e-6, f"z_0 is still zeros (max={z0_magnitude})"

    print(f"✓ z changes across all {config.reasoner_iterations} iterations")


def test_different_inputs_produce_different_z():
    """Different inputs must produce different z* — reasoner isn't collapsed."""
    config = TINY_CONFIG
    model = RGEN(config)
    model.eval()

    batch, seq_len = 1, 16

    # Two very different inputs
    input_a = torch.randint(0, config.vocab_size // 2, (batch, seq_len))
    input_b = torch.randint(config.vocab_size // 2, config.vocab_size, (batch, seq_len))

    with torch.no_grad():
        x_a = model.embedding(input_a)
        x_b = model.embedding(input_b)
        z_a = model.reasoner(x_a)
        z_b = model.reasoner(x_b)

    diff = (z_a - z_b).abs().max().item()
    assert diff > 1e-4, f"z* is nearly identical for different inputs (max diff={diff})"
    print(f"✓ Different inputs produce different z* (max diff={diff:.6f})")


def test_weight_sharing():
    """Verify layers are reused (weight sharing) — same object across iterations."""
    config = TINY_CONFIG
    model = RGEN(config)

    # The reasoner should have exactly reasoner_layers layers, not K * reasoner_layers
    n_layers = len(model.reasoner.layers)
    assert n_layers == config.reasoner_layers, \
        f"Expected {config.reasoner_layers} layers (weight-shared), got {n_layers}"
    print(f"✓ Weight sharing confirmed: {n_layers} layers reused across {config.reasoner_iterations} iterations")


def test_z_star_shape():
    """z* must have the same shape as the input embedding."""
    config = TINY_CONFIG
    model = RGEN(config)
    model.eval()

    batch, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))

    with torch.no_grad():
        x = model.embedding(input_ids)
        z_star = model.reasoner(x)

    assert z_star.shape == x.shape, \
        f"z* shape {z_star.shape} != input shape {x.shape}"
    assert not torch.isnan(z_star).any(), "NaN in z*"
    print(f"✓ z* shape matches input: {z_star.shape}")


if __name__ == "__main__":
    test_z_changes_each_iteration()
    test_different_inputs_produce_different_z()
    test_weight_sharing()
    test_z_star_shape()
    print("\nAll reasoner tests passed!")
