"""Tiny ViT encoder shared by I-JEPA's context and target encoders.

The context and target encoders in I-JEPA have the SAME architecture (the
target encoder is just an EMA copy of the context encoder's weights). This
module only builds that shared architecture; the EMA copying logic is a
separate, later piece.

Config picked for CIFAR-10 (32x32 images) on an 8GB GPU: patch_size=4 gives
an 8x8=64 patch grid, matching the grid already used in masking.py. embed_dim
192 / depth 6 / 3 heads is a standard "tiny ViT" scale (c.f. DeiT-Tiny), sized
down further in depth since CIFAR-10 is far simpler than ImageNet.
"""

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    Splits an image into non-overlapping patches and linearly projects each one.

    Implemented as a single strided Conv2d, which is mathematically the same
    as flattening each patch and applying a linear layer, but faster.
    """

    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int) -> None:
        """Initializes the patch embedding layer.

        Args:
            img_size: Height and width of the (square) input image, in pixels.
            patch_size: Height and width of each (square) patch, in pixels.
            in_chans: Number of input image channels (3 for RGB).
            embed_dim: Output embedding dimension per patch.

        Raises:
            ValueError: If img_size is not evenly divisible by patch_size.
        """
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size {img_size} not divisible by patch_size {patch_size}")
        self.grid_h = img_size // patch_size
        self.grid_w = img_size // patch_size
        self.num_patches = self.grid_h * self.grid_w
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Patchifies and embeds a batch of images.

        Args:
            x: Image batch of shape (batch, in_chans, img_size, img_size).

        Returns:
            Patch embeddings of shape (batch, num_patches, embed_dim), in
            row-major patch order (same convention as masking.py).
        """
        x = self.proj(x)  # (batch, embed_dim, grid_h, grid_w)
        x = x.flatten(2).transpose(1, 2)  # (batch, num_patches, embed_dim)
        return x


class TransformerBlock(nn.Module):
    """One pre-norm transformer block: self-attention + MLP, each with a residual."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        """Initializes the block.

        Args:
            embed_dim: Token embedding dimension.
            num_heads: Number of self-attention heads. Must divide embed_dim.
            mlp_ratio: Width of the MLP's hidden layer, as a multiple of embed_dim.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies self-attention then the MLP, each wrapped in a residual connection.

        Args:
            x: Token sequence of shape (batch, seq_len, embed_dim).

        Returns:
            Transformed token sequence, same shape as the input.
        """
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """
    Tiny Vision Transformer encoder, with no [CLS] token (per the I-JEPA paper).

    Used as both the context encoder and (an EMA copy of it as) the target
    encoder. Supports running on only a subset of patches via `patch_indices`,
    which is how the context encoder sees only the sparse, unmasked patches
    (not the full grid) while the target encoder sees all of them.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
    ) -> None:
        """Initializes the encoder.

        Args:
            img_size: Height and width of the (square) input image, in pixels.
            patch_size: Height and width of each (square) patch, in pixels.
            in_chans: Number of input image channels.
            embed_dim: Token embedding dimension throughout the encoder.
            depth: Number of transformer blocks.
            num_heads: Number of self-attention heads per block.
            mlp_ratio: Width of each block's MLP hidden layer, as a multiple of embed_dim.
        """
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, patch_indices: torch.Tensor | None = None) -> torch.Tensor:
        """
        Encodes an image batch, optionally restricted to a subset of patches.

        Args:
            x: Image batch of shape (batch, in_chans, img_size, img_size).
            patch_indices: Optional 1D tensor of patch indices to keep (same
                indices applied to every image in the batch, matching the
                paper's batching convention from masking.py). If None, all
                patches are used (target-encoder usage).

        Returns:
            Encoded tokens of shape (batch, num_kept_patches, embed_dim).
        """
        tokens = self.patch_embed(x) + self.pos_embed
        if patch_indices is not None:
            tokens = tokens[:, patch_indices, :]
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    from masking import sample_masks
    import random

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ViTEncoder().to(device)

    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"device: {device}")
    print(f"total params: {num_params:,} ({num_params * 4 / 1e6:.1f} MB as float32)\n")

    batch_size = 8
    images = torch.randn(batch_size, 3, 32, 32, device=device)

    # Target-encoder style: full image, all 64 patches.
    target_out = encoder(images)
    print(f"full forward (target-encoder style): {tuple(target_out.shape)}")
    assert target_out.shape == (batch_size, 64, 192)

    # Context-encoder style: only the sparse visible patches from sample_masks.
    rng = random.Random(0)
    context_indices, target_blocks = sample_masks(8, 8, rng=rng)
    context_idx_tensor = torch.tensor(context_indices, device=device)
    context_out = encoder(images, patch_indices=context_idx_tensor)
    print(f"sparse forward (context-encoder style): {tuple(context_out.shape)}")
    assert context_out.shape == (batch_size, len(context_indices), 192)

    print("\nall shape checks passed")
