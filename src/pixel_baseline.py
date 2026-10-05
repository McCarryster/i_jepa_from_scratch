"""MAE-style pixel-reconstruction baseline - row 2 of the project's comparison table.

Same masking scheme and context encoder architecture as I-JEPA, but the
decoder reconstructs raw target PIXELS instead of target latent embeddings,
and there's no target encoder or EMA at all - just one encoder plus a
decoder, trained end-to-end by backprop against pixel values. This isolates
the one variable the project's central question is about: predicting in
latent space vs. predicting in pixel space, with everything else (masking,
encoder architecture, optimizer schedule) held identical to train.py.

Loss is plain per-pixel MSE (mean over batch, patch, and pixel dims equally),
the standard choice for pixel reconstruction (e.g. the MAE paper) - not made
to match ijepa_loss's aggregation, since pixel-space and latent-space losses
aren't meant to be compared by magnitude anyway, only by downstream linear
probe accuracy.
"""

import random
from typing import Any

import torch
from torch import nn

from encoder import ViTEncoder
from masking import sample_masks
from predictor import Predictor
from train import GRID_H, GRID_W, get_cifar10_loader, lr_at_step


def patchify_pixels(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Slices an image batch into flattened raw-pixel patches, row-major order.

    Uses the same row-major patch ordering as encoder.py's PatchEmbed (patch
    index = row * grid_w + col), so patch indices from masking.py apply to
    this output directly, with no reordering needed.

    Args:
        images: Image batch, shape (batch, channels, height, width).
        patch_size: Height/width of each square patch, in pixels.

    Returns:
        Flattened patches, shape (batch, num_patches, channels * patch_size^2).
    """
    batch, channels, height, width = images.shape
    grid_h = height // patch_size
    grid_w = width // patch_size
    x = images.reshape(batch, channels, grid_h, patch_size, grid_w, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)
    return x.reshape(batch, grid_h * grid_w, channels * patch_size * patch_size)


def train_pixel_baseline(
    num_steps: int = 20,
    batch_size: int = 64,
    warmup_fraction: float = 0.1,
    log_every: int = 5,
    seed: int = 0,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Trains the MAE-style pixel-reconstruction baseline.

    Args:
        num_steps: Number of training steps (batches) to run.
        batch_size: Batch size.
        warmup_fraction: Fraction of num_steps spent on LR warmup.
        log_every: Print loss every this many steps.
        seed: Random seed for mask sampling.
        device: Device to train on. Defaults to CUDA if available.

    Returns:
        A dict with keys "encoder" (the trained encoder - there is only one,
        no separate target encoder for this baseline), "decoder", "losses",
        and "max_param_change" (sanity signal that training moved the weights).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(seed)
    patch_size = 32 // GRID_W

    loader = get_cifar10_loader(batch_size)

    encoder = ViTEncoder().to(device)
    patch_dim = 3 * patch_size * patch_size
    decoder = Predictor(
        encoder_dim=192, num_patches=GRID_H * GRID_W, output_dim=patch_dim
    ).to(device)

    initial_state = {k: v.clone() for k, v in encoder.state_dict().items()}

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4
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
        predictions = decoder(context_tokens, context_idx_t, flat_target_idx)

        target_pixels = patchify_pixels(images, patch_size)[:, flat_target_idx, :]
        loss = nn.functional.mse_loss(predictions, target_pixels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if step % log_every == 0 or step == num_steps - 1:
            print(f"step {step:4d}/{num_steps}  loss={loss.item():.4f}  lr={lr:.2e}")

    max_param_change = max(
        (encoder.state_dict()[k] - v).abs().max().item() for k, v in initial_state.items()
    )
    print(f"max abs parameter change from init: {max_param_change:.6f}")

    return {
        "encoder": encoder,
        "decoder": decoder,
        "losses": losses,
        "max_param_change": max_param_change,
    }


if __name__ == "__main__":
    import math

    # 1. patchify_pixels ordering check against a known, hand-computed example.
    # 1x1x4x4 image, patch_size=2 -> 2x2=4 patches, each 1*2*2=4 values.
    image = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    patches = patchify_pixels(image, patch_size=2)
    assert patches.shape == (1, 4, 4)
    # Patch 0 = top-left 2x2 block: rows 0-1, cols 0-1 -> pixels 0,1,4,5.
    assert torch.equal(patches[0, 0], torch.tensor([0.0, 1.0, 4.0, 5.0]))
    # Patch 1 = top-right 2x2 block: rows 0-1, cols 2-3 -> pixels 2,3,6,7.
    assert torch.equal(patches[0, 1], torch.tensor([2.0, 3.0, 6.0, 7.0]))
    # Patch 2 = bottom-left: rows 2-3, cols 0-1 -> pixels 8,9,12,13.
    assert torch.equal(patches[0, 2], torch.tensor([8.0, 9.0, 12.0, 13.0]))
    print("check 1 passed: patchify_pixels ordering matches hand-computed patches")

    # 2. Full training smoke test.
    result = train_pixel_baseline(num_steps=20, batch_size=64, log_every=5)
    losses = result["losses"]
    assert len(losses) == 20
    assert all(math.isfinite(loss) for loss in losses), "non-finite loss encountered"
    print("check 2 passed: ran 20 steps, all losses finite")

    assert result["max_param_change"] > 0, "encoder weights never changed"
    print(f"check 3 passed: encoder weights changed "
          f"(max abs change {result['max_param_change']:.6f})")

    print("\nall checks passed")
