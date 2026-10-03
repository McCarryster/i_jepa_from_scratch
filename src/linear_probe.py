"""Linear probe evaluation: frozen encoder + average-pooled features + linear classifier.

Per this project's notes, evaluation reads from the TARGET encoder for an
I-JEPA run (not the context encoder) - pass whichever encoder should be
probed; for a trained I-JEPA run that's train.py's result["target_encoder"].

Protocol: extract average-pooled patch features once for the whole dataset
(encoder fully frozen, no augmentation), then fit a plain linear classifier
on top via cross-entropy. This is deliberately the SAME procedure for every
row of the project's comparison table (random-init floor, pixel-reconstruction
baseline, I-JEPA with/without EMA) so accuracy differences come only from the
encoder, not from differences in how the probe itself was trained.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train import get_cifar10_loader


@torch.no_grad()
def extract_features(
    encoder: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extracts average-pooled patch features for an entire dataset.

    Args:
        encoder: A frozen encoder (e.g. ViTEncoder). Run in eval mode, not modified.
        loader: DataLoader yielding (images, labels) batches.
        device: Device to run the encoder on.

    Returns:
        A (features, labels) tuple: features has shape (dataset_size, embed_dim),
        labels has shape (dataset_size,).
    """
    encoder.eval()
    all_features = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device)
        tokens = encoder(images)  # (batch, num_patches, embed_dim); full image, no masking
        pooled = tokens.mean(dim=1)  # average-pool over patches; no [CLS] token to exclude
        all_features.append(pooled.cpu())
        all_labels.append(labels)
    return torch.cat(all_features), torch.cat(all_labels)


def train_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int = 10,
    num_epochs: int = 20,
    lr: float = 1e-2,
    batch_size: int = 256,
    device: torch.device | None = None,
) -> float:
    """Trains a linear classifier on precomputed frozen features.

    Args:
        train_features: Training features, shape (num_train, dim).
        train_labels: Training labels, shape (num_train,).
        test_features: Test features, shape (num_test, dim).
        test_labels: Test labels, shape (num_test,).
        num_classes: Number of output classes.
        num_epochs: Number of training epochs over the features.
        lr: Learning rate for the linear classifier.
        batch_size: Minibatch size for probe training.
        device: Device to train the probe on. Defaults to CUDA if available.

    Returns:
        Top-1 test accuracy, as a float in [0, 1].
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = train_features.shape[1]
    probe = nn.Linear(dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)

    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)

    num_train = train_features.shape[0]
    for _ in range(num_epochs):
        perm = torch.randperm(num_train, device=device)
        for start in range(0, num_train, batch_size):
            idx = perm[start : start + batch_size]
            logits = probe(train_features[idx])
            loss = nn.functional.cross_entropy(logits, train_labels[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        test_preds = probe(test_features).argmax(dim=1)
        accuracy = (test_preds == test_labels).float().mean().item()

    return accuracy


def linear_probe_accuracy(
    encoder: nn.Module,
    device: torch.device | None = None,
    batch_size: int = 256,
    probe_epochs: int = 20,
    probe_lr: float = 1e-2,
) -> float:
    """Runs the full linear-probe protocol: extract features, fit probe, return test accuracy.

    Args:
        encoder: Frozen encoder to evaluate.
        device: Device to run on. Defaults to CUDA if available.
        batch_size: Batch size for feature extraction.
        probe_epochs: Number of epochs to train the linear probe.
        probe_lr: Learning rate for the linear probe.

    Returns:
        Top-1 test accuracy on CIFAR-10, as a float in [0, 1].
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)

    train_loader = get_cifar10_loader(batch_size, train=True)
    test_loader = get_cifar10_loader(batch_size, train=False)

    train_features, train_labels = extract_features(encoder, train_loader, device)
    test_features, test_labels = extract_features(encoder, test_loader, device)

    return train_linear_probe(
        train_features,
        train_labels,
        test_features,
        test_labels,
        num_epochs=probe_epochs,
        lr=probe_lr,
        batch_size=batch_size,
        device=device,
    )


if __name__ == "__main__":
    from encoder import ViTEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    print("evaluating a RANDOM-INIT encoder (expected: low accuracy, near chance)")
    random_encoder = ViTEncoder().to(device)
    accuracy = linear_probe_accuracy(random_encoder, device=device, probe_epochs=20)
    print(f"random-init linear probe accuracy: {accuracy * 100:.2f}%")

    assert 0.0 <= accuracy <= 1.0
    assert accuracy < 0.4, f"random-init accuracy suspiciously high ({accuracy * 100:.1f}%) - check for a bug"
    print("check passed: random-init accuracy is low, as expected for an untrained encoder")

    print("\nall checks passed")
