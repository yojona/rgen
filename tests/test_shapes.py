"""Validate tensor shapes through the full RGEN forward pass."""

import torch
from rgen.config import TINY_CONFIG
from rgen.model import RGEN


def test_forward_shapes():
    config = TINY_CONFIG
    model = RGEN(config)
    model.eval()

    batch, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))

    with torch.no_grad():
        logits = model(input_ids)

    assert logits.shape == (batch, seq_len, config.vocab_size), \
        f"Expected logits shape {(batch, seq_len, config.vocab_size)}, got {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN detected in logits"
    assert not torch.isinf(logits).any(), "Inf detected in logits"
    print(f"✓ Forward pass shapes OK: logits {logits.shape}")


def test_reasoner_shapes():
    config = TINY_CONFIG
    model = RGEN(config)
    model.eval()

    batch, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))

    with torch.no_grad():
        x = model.embedding(input_ids)
        z_star = model.reasoner(x)

    assert z_star.shape == (batch, seq_len, config.d_model), \
        f"Expected z_star shape {(batch, seq_len, config.d_model)}, got {z_star.shape}"
    assert not torch.isnan(z_star).any(), "NaN detected in z_star"
    print(f"✓ Reasoner shapes OK: z_star {z_star.shape}")


def test_no_nan_activations():
    config = TINY_CONFIG
    model = RGEN(config)
    model.eval()

    batch, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))

    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activations[name] = output
        return hook

    for name, module in model.named_modules():
        module.register_forward_hook(hook_fn(name))

    with torch.no_grad():
        model(input_ids)

    nan_modules = [name for name, act in activations.items() if torch.isnan(act).any()]
    assert len(nan_modules) == 0, f"NaN found in: {nan_modules}"
    print(f"✓ No NaN in any of {len(activations)} activations")


if __name__ == "__main__":
    test_forward_shapes()
    test_reasoner_shapes()
    test_no_nan_activations()
    print("\nAll shape tests passed!")
