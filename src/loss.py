"""I-JEPA loss: mean squared L2 distance between predicted and target embeddings.

Per-patch distance is D(pred, target) = ||pred - target||_2^2 (squared, not
raw Euclidean norm - squared L2 gives smoother gradients, standard for this
kind of latent-prediction loss). The final loss is the mean of D over every
(batch element, target patch) pair, pooling all target blocks together into
one flat set of patches rather than weighting blocks separately.

Note: the paper's exact block-weighting (sum of 4 blocks' means vs. one
global mean over all blocks combined) is ambiguous between phrasings of it
floating around; this file picks the simplest, standard one (global mean).
Swap to per-block averaging later if exact paper replication turns out to
matter for a specific claim made on camera.
"""

import torch


def ijepa_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Computes the mean squared L2 distance between predicted and target embeddings.

    Args:
        predictions: Predicted patch embeddings, shape (batch, num_target, dim).
        targets: Target patch embeddings, same shape as predictions. Must come
            from the target encoder with no gradient tracking (see ema.py -
            its parameters already have requires_grad=False, so this holds by
            construction as long as targets is produced by that encoder).

    Returns:
        A scalar tensor: the mean squared L2 distance, averaged over the
        batch dimension and every target patch.

    Raises:
        ValueError: If predictions and targets have different shapes.
    """
    if predictions.shape != targets.shape:
        raise ValueError(
            f"shape mismatch: predictions {tuple(predictions.shape)} "
            f"vs targets {tuple(targets.shape)}"
        )
    squared_l2_per_patch = (predictions - targets).pow(2).sum(dim=-1)
    return squared_l2_per_patch.mean()


if __name__ == "__main__":
    import random
    import sys

    sys.path.insert(0, ".")
    from ema import make_target_encoder
    from encoder import ViTEncoder
    from masking import sample_masks
    from predictor import Predictor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # 1. Identical embeddings -> zero loss.
    same = torch.randn(4, 10, 32, device=device)
    assert ijepa_loss(same, same).item() == 0.0
    print("check 1 passed: identical predictions/targets give zero loss")

    # 2. Known constant offset -> loss equals that offset's squared L2 norm.
    targets = torch.zeros(1, 1, 4, device=device)
    predictions = torch.tensor([[[3.0, 4.0, 0.0, 0.0]]], device=device)  # ||.||^2 = 25
    loss = ijepa_loss(predictions, targets)
    assert abs(loss.item() - 25.0) < 1e-6
    print(f"check 2 passed: known 3-4-0-0 offset gives loss {loss.item()} (expected 25.0)")

    # 3. Shape mismatch raises.
    try:
        ijepa_loss(torch.randn(2, 3, 4), torch.randn(2, 3, 5))
        raise AssertionError("expected ValueError on shape mismatch")
    except ValueError:
        print("check 3 passed: shape mismatch raises ValueError")

    # 4. End-to-end: stop-gradient actually holds through the real pipeline,
    #    not just by convention - targets produced by the EMA target encoder
    #    must carry no gradient back to it.
    context_encoder = ViTEncoder().to(device)
    target_encoder = make_target_encoder(context_encoder)
    predictor = Predictor(encoder_dim=192, num_patches=64).to(device)

    images = torch.randn(4, 3, 32, 32, device=device)
    rng = random.Random(0)
    context_indices, target_blocks = sample_masks(8, 8, rng=rng)
    context_idx_t = torch.tensor(context_indices, device=device)
    flat_target_idx = torch.tensor(
        [idx for block in target_blocks for idx in block], device=device
    )

    context_tokens = context_encoder(images, patch_indices=context_idx_t)
    target_tokens_full = target_encoder(images)  # full image, all 64 patches
    target_tokens = target_tokens_full[:, flat_target_idx, :]

    predictions = predictor(context_tokens, context_idx_t, flat_target_idx)
    loss = ijepa_loss(predictions, target_tokens)

    assert not target_tokens.requires_grad, "target tokens must not require grad"
    loss.backward()
    context_grad = sum(
        p.grad.abs().sum().item() for p in context_encoder.parameters() if p.grad is not None
    )
    target_grad_exists = any(p.grad is not None for p in target_encoder.parameters())
    assert context_grad > 0, "no gradient reached the context encoder"
    assert not target_grad_exists, "gradient leaked into the frozen target encoder"
    print(f"check 4 passed: loss={loss.item():.4f}, gradient reaches context encoder "
          f"(abs sum {context_grad:.4f}), none leaks into target encoder")

    print("\nall checks passed")
