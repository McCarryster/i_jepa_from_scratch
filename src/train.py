"""I-JEPA pretraining loop: context encoder + EMA target encoder + predictor + loss.

Simplification from the paper: one random mask (context block + 4 target
blocks) is sampled PER BATCH and shared by every image in that batch, rather
than the paper's per-image random positions with only batch-uniform sizes.
This matches encoder.py's sparse-gather design, which takes a single index
tensor applied to every image in the batch. Trades some within-batch mask
diversity for a much simpler implementation - acceptable at this scale since
many different masks are still seen across steps/epochs.

Per the paper's eval protocol (and this project's own notes), linear-probe
evaluation reads from the TARGET encoder, not the context encoder - so this
module returns and checkpoints both, not just the one trained by backprop.
"""

import math
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ema import ema_momentum, make_target_encoder, update_target_encoder
from encoder import ViTEncoder
from loss import ijepa_loss
from masking import sample_masks
from predictor import Predictor

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
GRID_H = 8
GRID_W = 8
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def lr_at_step(
    step: int,
    total_steps: int,
    warmup_steps: int,
    start_lr: float = 1e-4,
    peak_lr: float = 1e-3,
    end_lr: float = 1e-6,
) -> float:
    """Computes the learning rate for a step: linear warmup then cosine decay.

    Args:
        step: Current training step, 0-indexed.
        total_steps: Total number of training steps.
        warmup_steps: Number of steps spent ramping linearly from start_lr to peak_lr.
        start_lr: Learning rate at step 0.
        peak_lr: Learning rate at the end of warmup.
        end_lr: Learning rate at the final step (cosine decay target).

    Returns:
        The learning rate for this step.
    """
    if step < warmup_steps:
        return start_lr + (peak_lr - start_lr) * step / warmup_steps
    progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
    return end_lr + 0.5 * (peak_lr - end_lr) * (1 + math.cos(math.pi * progress))


def weight_decay_at_step(step: int, total_steps: int, start: float = 0.04, end: float = 0.4) -> float:
    """Computes the weight decay for a step via linear ramp (paper's linear-probe setting).

    Args:
        step: Current training step, 0-indexed.
        total_steps: Total number of training steps.
        start: Weight decay at step 0.
        end: Weight decay at the final step.

    Returns:
        The weight decay for this step.
    """
    progress = min(step / total_steps, 1.0)
    return start + (end - start) * progress


def get_cifar10_loader(batch_size: int, train: bool = True) -> DataLoader:
    """Builds a CIFAR-10 DataLoader with no augmentation (I-JEPA pretraining uses none).

    Args:
        batch_size: Batch size.
        train: Whether to load the train split (vs. test split).

    Returns:
        A DataLoader yielding (images, labels) batches, images normalized
        but not augmented.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    )
    dataset = datasets.CIFAR10(root=str(DATA_DIR), train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2, drop_last=train)


def train(
    num_steps: int = 20,
    batch_size: int = 64,
    warmup_fraction: float = 0.1,
    checkpoint_path: str | None = None,
    log_every: int = 5,
    seed: int = 0,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Runs the I-JEPA pretraining loop.

    Args:
        num_steps: Number of training steps (batches) to run.
        batch_size: Batch size.
        warmup_fraction: Fraction of num_steps spent on LR warmup.
        checkpoint_path: If given, save both encoders' state_dicts here.
        log_every: Print loss every this many steps.
        seed: Random seed for mask sampling.
        device: Device to train on. Defaults to CUDA if available.

    Returns:
        A dict with keys "context_encoder", "target_encoder", "predictor"
        (the trained modules), "losses" (per-step loss list), and
        "max_param_change" (largest absolute change in any context encoder
        parameter from its initial value, a quick sanity signal that
        training actually moved the weights).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(seed)

    loader = get_cifar10_loader(batch_size)

    context_encoder = ViTEncoder().to(device)
    target_encoder = make_target_encoder(context_encoder)
    predictor = Predictor(encoder_dim=192, num_patches=GRID_H * GRID_W).to(device)

    initial_state = {k: v.clone() for k, v in context_encoder.state_dict().items()}

    optimizer = torch.optim.AdamW(
        list(context_encoder.parameters()) + list(predictor.parameters()), lr=1e-4
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
        weight_decay = weight_decay_at_step(step, num_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
            group["weight_decay"] = weight_decay

        context_indices, target_blocks = sample_masks(GRID_H, GRID_W, rng=rng)
        context_idx_t = torch.tensor(context_indices, device=device)
        flat_target_idx = torch.tensor(
            [idx for block in target_blocks for idx in block], device=device
        )

        context_tokens = context_encoder(images, patch_indices=context_idx_t)
        with torch.no_grad():
            target_tokens_full = target_encoder(images)
        target_tokens = target_tokens_full[:, flat_target_idx, :]

        predictions = predictor(context_tokens, context_idx_t, flat_target_idx)
        loss = ijepa_loss(predictions, target_tokens)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        momentum = ema_momentum(step, num_steps)
        update_target_encoder(target_encoder, context_encoder, momentum)

        losses.append(loss.item())
        if step % log_every == 0 or step == num_steps - 1:
            print(
                f"step {step:4d}/{num_steps}  loss={loss.item():.4f}  "
                f"lr={lr:.2e}  wd={weight_decay:.3f}  momentum={momentum:.5f}"
            )

    max_param_change = max(
        (context_encoder.state_dict()[k] - v).abs().max().item()
        for k, v in initial_state.items()
    )
    print(f"max abs parameter change from init: {max_param_change:.6f}")

    if checkpoint_path is not None:
        torch.save(
            {
                "context_encoder": context_encoder.state_dict(),
                "target_encoder": target_encoder.state_dict(),
            },
            checkpoint_path,
        )
        print(f"saved checkpoint to {checkpoint_path}")

    return {
        "context_encoder": context_encoder,
        "target_encoder": target_encoder,
        "predictor": predictor,
        "losses": losses,
        "max_param_change": max_param_change,
    }


if __name__ == "__main__":
    # Quick smoke test: confirms the full pipeline trains without crashing or
    # producing NaNs. NOT a real pretraining run - num_steps here is tiny.
    # For a real run, call train(num_steps=<thousands>, batch_size=256, ...).

    assert lr_at_step(0, 100, 10) == 1e-4
    assert abs(lr_at_step(100, 100, 10) - 1e-6) < 1e-9
    print("check 1 passed: lr schedule hits start_lr at step 0, end_lr at final step")

    assert weight_decay_at_step(0, 100) == 0.04
    assert abs(weight_decay_at_step(100, 100) - 0.4) < 1e-9
    print("check 2 passed: weight decay schedule hits 0.04 at step 0, 0.4 at final step")

    result = train(num_steps=20, batch_size=64, log_every=5)

    losses = result["losses"]
    assert len(losses) == 20
    assert all(math.isfinite(loss) for loss in losses), "non-finite loss encountered"
    print("check 3 passed: ran 20 steps, all losses finite")

    assert result["max_param_change"] > 0, "context encoder weights never changed"
    print(f"check 4 passed: context encoder weights changed "
          f"(max abs change {result['max_param_change']:.6f})")

    print("\nall checks passed")
