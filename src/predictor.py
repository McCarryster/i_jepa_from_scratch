"""Narrow predictor for I-JEPA: context tokens + mask tokens -> target predictions.

Given the context encoder's output tokens and the patch indices of the
positions to predict, builds a sequence of [projected context tokens, mask
tokens], runs it through a small transformer, and returns predictions for
just the mask-token positions, projected back up to the encoder's dimension.

A mask token only depends on ITS OWN patch position (shared learnable vector
+ that position's embedding) - it has no notion of "which target block" it
belongs to. So this module takes one flat tensor of target patch indices;
if the caller has 4 target blocks (possibly overlapping), it concatenates
their indices before calling forward() and splits the output back into
per-block chunks afterward (for the loss, built separately).

Width/depth picked narrower and shallower than the context encoder
(predictor_dim=64 vs encoder embed_dim=192, depth=4 vs encoder depth=6),
mirroring the paper's bottleneck predictor: it should do real spatial
reasoning (hence >1 layer, unlike a bare linear probe) without being able to
carry as much representational capacity as the encoder itself.
"""

import torch
import torch.nn as nn

from encoder import TransformerBlock


class Predictor(nn.Module):
    """Predicts target patch embeddings from visible context patch embeddings."""

    def __init__(
        self,
        encoder_dim: int,
        num_patches: int,
        predictor_dim: int = 64,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        output_dim: int | None = None,
    ) -> None:
        """Initializes the predictor.

        Args:
            encoder_dim: Embedding dimension used by the context/target encoders.
            num_patches: Total number of patches in the grid (for the predictor's
                own positional embedding table).
            predictor_dim: Internal embedding dimension of the predictor.
            depth: Number of transformer blocks.
            num_heads: Number of self-attention heads per block.
            mlp_ratio: Width of each block's MLP hidden layer, as a multiple of predictor_dim.
            output_dim: Dimension to project predictions back up to. Defaults to
                encoder_dim (I-JEPA's own latent-prediction case). Pass a different
                value to reuse this same architecture as a pixel decoder (e.g.
                channels * patch_size^2) instead of a latent predictor.
        """
        super().__init__()
        output_dim = output_dim if output_dim is not None else encoder_dim
        self.context_proj = nn.Linear(encoder_dim, predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(predictor_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(predictor_dim)
        self.out_proj = nn.Linear(predictor_dim, output_dim)

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts embeddings at target_indices from context_tokens.

        Args:
            context_tokens: Context encoder output, shape (batch, num_context, encoder_dim).
            context_indices: Patch indices the context tokens came from, shape
                (num_context,). Used to look up their positional embeddings here
                (the predictor has its own positional embedding table, separate
                from the encoder's).
            target_indices: Patch indices to predict, shape (num_target,). May
                contain duplicates if target blocks overlap - each occurrence
                gets its own predicted output, matching the paper's loss, which
                sums over every (block, position) pair rather than deduplicating.

        Returns:
            Predicted embeddings at the target positions, shape
            (batch, num_target, output_dim) (output_dim set in __init__).
        """
        batch_size = context_tokens.shape[0]

        context = self.context_proj(context_tokens) + self.pos_embed[:, context_indices, :]

        mask_tokens = self.mask_token.expand(batch_size, target_indices.shape[0], -1)
        mask_tokens = mask_tokens + self.pos_embed[:, target_indices, :]

        sequence = torch.cat([context, mask_tokens], dim=1)
        for block in self.blocks:
            sequence = block(sequence)
        sequence = self.norm(sequence)
        sequence = self.out_proj(sequence)

        num_context = context_tokens.shape[1]
        return sequence[:, num_context:, :]


if __name__ == "__main__":
    import random
    import sys

    sys.path.insert(0, ".")
    from encoder import ViTEncoder
    from masking import sample_masks

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    encoder = ViTEncoder().to(device)
    predictor = Predictor(encoder_dim=192, num_patches=64).to(device)

    batch_size = 8
    images = torch.randn(batch_size, 3, 32, 32, device=device)

    rng = random.Random(0)
    context_indices, target_blocks = sample_masks(8, 8, rng=rng)
    context_idx_t = torch.tensor(context_indices, device=device)

    context_tokens = encoder(images, patch_indices=context_idx_t)
    print(f"context tokens: {tuple(context_tokens.shape)}")

    # Concatenate all 4 target blocks into one flat index tensor.
    all_target_indices = [idx for block in target_blocks for idx in block]
    target_idx_t = torch.tensor(all_target_indices, device=device)
    print(f"flat target indices: {len(all_target_indices)} "
          f"(sum of block sizes: {[len(b) for b in target_blocks]})")

    predictions = predictor(context_tokens, context_idx_t, target_idx_t)
    print(f"predictions: {tuple(predictions.shape)}")
    assert predictions.shape == (batch_size, len(all_target_indices), 192)

    # Splitting back into per-block predictions (how the loss will consume this).
    split_sizes = [len(b) for b in target_blocks]
    per_block_preds = torch.split(predictions, split_sizes, dim=1)
    for i, block_pred in enumerate(per_block_preds):
        assert block_pred.shape == (batch_size, len(target_blocks[i]), 192)
    print(f"split into {len(per_block_preds)} per-block predictions, shapes match")

    # Gradient sanity check: loss on predictions should flow back into the
    # context encoder's parameters (predictor must actually be trainable
    # end-to-end with the context encoder).
    loss = predictions.pow(2).mean()
    loss.backward()
    grad_norm = sum(p.grad.abs().sum().item() for p in encoder.parameters() if p.grad is not None)
    assert grad_norm > 0, "no gradient reached the context encoder"
    print(f"gradient reached context encoder (abs sum = {grad_norm:.4f})")

    print("\nall checks passed")
