"""I-JEPA ablation with EMA and stop-gradient removed - row 3 of the comparison table.

Same masking scheme, encoder architecture, and predictor as train.py's full
I-JEPA, but with the one change the paper says prevents collapse removed: a
SINGLE shared encoder computes both the context tokens and the "target"
tokens (no separate EMA target encoder), and the target branch is NOT
wrapped in torch.no_grad() (no stop-gradient). Gradients from the L2 loss
flow into the encoder through both branches at once.

With nothing stopping it, the easiest way to drive the loss toward zero is
for the encoder to collapse to a constant output regardless of its input -
the predictor then trivially "predicts" that same constant. Expected result:
the encoder's linear-probe accuracy should fall back to roughly the
random-init floor, since a collapsed encoder carries no class information.
"""

import random
from typing import Any

import torch

from encoder import ViTEncoder
from loss import ijepa_loss
from masking import sample_masks
from predictor import Predictor
from train import GRID_H, GRID_W, get_cifar10_loader, lr_at_step


def train_no_ema(
    num_steps: int = 20,
    batch_size: int = 64,
    warmup_fraction: float = 0.1,
    log_every: int = 5,
    seed: int = 0,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Trains I-JEPA with EMA/stop-gradient removed, to demonstrate collapse.

    Args:
        num_steps: Number of training steps (batches) to run.
        batch_size: Batch size.
        warmup_fraction: Fraction of num_steps spent on LR warmup.
        log_every: Print loss every this many steps.
        seed: Random seed for mask sampling.
        device: Device to train on. Defaults to CUDA if available.

    Returns:
        A dict with keys "encoder" (the single shared encoder, trained with
        no stop-gradient), "predictor", "losses", and "feature_std" (the
        average per-dimension standard deviation of pooled encoder outputs
        across a batch of different test images, measured after training -
        near zero indicates collapse: every image mapping to roughly the
        same vector regardless of its content).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(seed)

    loader = get_cifar10_loader(batch_size)

    encoder = ViTEncoder().to(device)
    predictor = Predictor(encoder_dim=192, num_patches=GRID_H * GRID_W).to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()), lr=1e-4
    )
    warmup_steps = max(int(num_steps * warmup_fraction), 1)

    data_iter = iter(loader)
    losses = []

    for step in range(num_steps):
        try:
            images, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, _ = next(data_iter)
        images = images.to(device)

        lr = lr_at_step(step, num_steps, warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        context_indices, target_blocks = sample_masks(GRID_H, GRID_W, rng=rng)
        context_idx_t = torch.tensor(context_indices, device=device)
        flat_target_idx = torch.tensor(
            [idx for block in target_blocks for idx in block], device=device
        )

        context_tokens = encoder(images, patch_indices=context_idx_t)
        # Same encoder, no torch.no_grad() - the one change from train.py's
        # stop-gradient EMA setup. Gradients flow into these shared weights
        # through both the context branch and this "target" branch at once.
        target_tokens_full = encoder(images)
        target_tokens = target_tokens_full[:, flat_target_idx, :]

        predictions = predictor(context_tokens, context_idx_t, flat_target_idx)
        loss = ijepa_loss(predictions, target_tokens)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if step % log_every == 0 or step == num_steps - 1:
            print(f"step {step:4d}/{num_steps}  loss={loss.item():.4f}  lr={lr:.2e}")

    with torch.no_grad():
        eval_images, _ = next(iter(get_cifar10_loader(batch_size, train=False)))
        eval_images = eval_images.to(device)
        pooled = encoder(eval_images).mean(dim=1)  # (batch, embed_dim), full image
        feature_std = pooled.std(dim=0).mean().item()  # spread across different images
    print(f"feature std across a batch of different images: {feature_std:.6f}")

    return {
        "encoder": encoder,
        "predictor": predictor,
        "losses": losses,
        "feature_std": feature_std,
    }


if __name__ == "__main__":
    import math

    torch.manual_seed(0)

    # Baseline for comparison: an UNTRAINED encoder's feature spread, so the
    # trained run's feature_std can be read relative to something.
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        random_encoder = ViTEncoder().to(device)
        probe_images, _ = next(iter(get_cifar10_loader(64, train=False)))
        probe_images = probe_images.to(device)
        random_pooled = random_encoder(probe_images).mean(dim=1)
        random_feature_std = random_pooled.std(dim=0).mean().item()
    print(f"untrained encoder feature std (reference point): {random_feature_std:.6f}\n")

    result = train_no_ema(num_steps=20, batch_size=64, log_every=5)

    losses = result["losses"]
    assert len(losses) == 20
    assert all(math.isfinite(loss) for loss in losses), "non-finite loss encountered"
    print("check 1 passed: ran 20 steps, all losses finite")

    print(
        f"\nfeature std: untrained={random_feature_std:.6f}, "
        f"after no-EMA training={result['feature_std']:.6f}"
    )
    print(
        "(only 20 steps here - a real collapse signal needs the full training "
        "run + linear probe, this is just confirming the mechanism runs end to end)"
    )

    print("\nall checks passed")
