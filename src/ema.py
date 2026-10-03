"""EMA (exponential moving average) target encoder logic for I-JEPA.

The target encoder is architecturally identical to the context encoder (see
encoder.py) - it starts as an exact copy of it, then is never touched by
backprop. After every training step, its weights are nudged slightly toward
the context encoder's current weights:

    target_weights <- momentum * target_weights + (1 - momentum) * context_weights

momentum starts at 0.996 and ramps linearly to 1.0 over training, so the
target encoder changes fast early on (when the context encoder is still far
from its own initialization) and barely changes near the end.
"""

import copy

import torch.nn as nn


def make_target_encoder(context_encoder: nn.Module) -> nn.Module:
    """Creates the target encoder as a frozen copy of the context encoder.

    Args:
        context_encoder: The encoder being trained by backprop.

    Returns:
        A deep copy of context_encoder, in eval mode, with every parameter's
        requires_grad set to False so autograd never tracks it.
    """
    target_encoder = copy.deepcopy(context_encoder)
    for param in target_encoder.parameters():
        param.requires_grad = False
    target_encoder.eval()
    return target_encoder


def ema_momentum(step: int, total_steps: int, start: float = 0.996, end: float = 1.0) -> float:
    """Computes the EMA momentum for the current training step via linear ramp.

    Args:
        step: Current training step, 0-indexed.
        total_steps: Total number of training steps.
        start: Momentum at step 0.
        end: Momentum at the final step.

    Returns:
        The momentum value for this step, linearly interpolated between
        start and end.

    Raises:
        ValueError: If total_steps is not positive.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    progress = min(step / total_steps, 1.0)
    return start + (end - start) * progress


def update_target_encoder(
    target_encoder: nn.Module, context_encoder: nn.Module, momentum: float
) -> None:
    """Updates the target encoder's weights in place via EMA of the context encoder.

    Args:
        target_encoder: The frozen target encoder to update (modified in place).
        context_encoder: The context encoder currently being trained by backprop.
        momentum: EMA momentum for this step, typically from ema_momentum().
    """
    target_params = dict(target_encoder.named_parameters())
    for name, context_param in context_encoder.named_parameters():
        target_param = target_params[name]
        target_param.data.mul_(momentum).add_(context_param.data, alpha=1 - momentum)


if __name__ == "__main__":
    import torch

    import sys

    sys.path.insert(0, ".")
    from encoder import ViTEncoder

    torch.manual_seed(0)
    context_encoder = ViTEncoder(depth=2)  # small depth, just for a fast self-test
    target_encoder = make_target_encoder(context_encoder)

    # 1. Target starts as an exact copy.
    for (name, c_param), (_, t_param) in zip(
        context_encoder.named_parameters(), target_encoder.named_parameters()
    ):
        assert torch.equal(c_param, t_param), f"{name} differs right after copy"
    print("check 1 passed: target starts identical to context")

    # 2. Target's parameters don't track gradients.
    assert all(not p.requires_grad for p in target_encoder.parameters())
    print("check 2 passed: target parameters have requires_grad=False")

    # 3. Momentum schedule ramps linearly from 0.996 to 1.0.
    total_steps = 1000
    assert abs(ema_momentum(0, total_steps) - 0.996) < 1e-9
    assert abs(ema_momentum(total_steps, total_steps) - 1.0) < 1e-9
    assert abs(ema_momentum(total_steps // 2, total_steps) - 0.998) < 1e-9
    print("check 3 passed: momentum schedule ramps 0.996 -> 1.0 linearly")

    # 4. After perturbing the context encoder (simulating a training step) and
    #    running one EMA update, the target should move PART WAY toward it,
    #    not all the way.
    with torch.no_grad():
        for param in context_encoder.parameters():
            param.add_(1.0)  # simulate a large gradient step

    momentum = 0.9  # exaggerated, away from 0.996, so the shift is easy to see
    first_param_name = next(iter(dict(context_encoder.named_parameters())))
    target_before = dict(target_encoder.named_parameters())[first_param_name].clone()
    context_after = dict(context_encoder.named_parameters())[first_param_name].clone()

    update_target_encoder(target_encoder, context_encoder, momentum)

    target_after = dict(target_encoder.named_parameters())[first_param_name]
    expected = momentum * target_before + (1 - momentum) * context_after
    assert torch.allclose(target_after, expected)
    assert not torch.equal(target_after, target_before), "target didn't move at all"
    assert not torch.equal(target_after, context_after), "target jumped all the way to context"
    print("check 4 passed: EMA update moves target partway toward context, matches formula")

    print("\nall checks passed")
