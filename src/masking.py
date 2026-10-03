"""I-JEPA multi-block masking, sampled directly on a patch-index grid.

No model/tensor code here on purpose - this is the part of I-JEPA most
likely to have off-by-one errors (block sizing, bounds, overlap removal),
so it gets tested in isolation first, against plain Python ints/sets.

Grid convention: patches are indexed row-major, 0 .. grid_h*grid_w - 1,
same as a flattened ViT patch sequence (no [CLS] token).
"""

import math
import random


def sample_block(
    grid_h: int,
    grid_w: int,
    scale_range: tuple[float, float],
    aspect_ratio_range: tuple[float, float],
    rng: random.Random,
    max_attempts: int = 10,
) -> tuple[int, int, int, int]:
    """
    Sample one rectangular block of patches on a grid_h x grid_w grid.

    Args:
        grid_h: Number of patch rows in the grid.
        grid_w: Number of patch columns in the grid.
        scale_range: (lo, hi) fraction of total grid area the block should cover.
            - Pass (0.15, 0.2) (used for the I-JEPA context block).
        aspect_ratio_range: (lo, hi) for block height/width ratio.
            - Pass (1.0, 1.0) for a fixed-square block (used for the I-JEPA context block).
        rng: Random source to sample from.
        max_attempts: Number of sampling attempts before giving up.

    Returns:
        A (top, left, h, w) tuple in patch coordinates: top-left corner and
        height/width of the block.

    Raises:
        RuntimeError: If no valid block could be sampled within max_attempts.
    """
    num_patches = grid_h * grid_w
    for _ in range(max_attempts):
        scale = rng.uniform(*scale_range)
        aspect_ratio = rng.uniform(*aspect_ratio_range)
        area = scale * num_patches
        h = int(round(math.sqrt(area * aspect_ratio)))
        w = int(round(math.sqrt(area / aspect_ratio)))
        h = min(max(h, 1), grid_h)
        w = min(max(w, 1), grid_w)
        top = rng.randint(0, grid_h - h)
        left = rng.randint(0, grid_w - w)
        return top, left, h, w
    raise RuntimeError("failed to sample a valid block")


def block_to_indices(top, left, h, w, grid_w):
    """Flatten a (top, left, h, w) rectangle into a set of row-major patch indices."""
    return {r * grid_w + c for r in range(top, top + h) for c in range(left, left + w)}


def sample_masks(
    grid_h,
    grid_w,
    num_target_blocks=4,
    target_scale=(0.15, 0.2),
    target_aspect=(0.75, 1.5),
    context_scale=(0.85, 1.0),
    context_aspect=(1.0, 1.0),
    rng=None,
    min_context_patches=1,
    max_resample_attempts=20,
):
    """Sample one context block + `num_target_blocks` target blocks for one image.

    Mirrors the I-JEPA paper: target blocks are sampled independently (may
    overlap each other), then a single square context block is sampled and
    any patches overlapping ANY target block are removed from it.

    Returns:
        context_indices: sorted list of patch indices visible to the context encoder.
        target_blocks: list of `num_target_blocks` sorted lists of patch indices,
            one per target block (these index into the target encoder's full
            output, not a separately-run sub-network).

    Note on batching: when this is wrapped in a data-loader collator, all
    images in a batch should share the same (h, w) for their context blocks
    and the same (h, w) for their target blocks (paper appendix), so the
    per-image index lists can be stacked. That batching concern is deliberately
    NOT handled here - this function only samples for a single image.
    """
    rng = rng or random.Random()

    for _ in range(max_resample_attempts):
        target_blocks = []
        target_union = set()
        for _ in range(num_target_blocks):
            top, left, h, w = sample_block(grid_h, grid_w, target_scale, target_aspect, rng)
            idx = block_to_indices(top, left, h, w, grid_w)
            target_blocks.append(idx)
            target_union |= idx

        top, left, h, w = sample_block(grid_h, grid_w, context_scale, context_aspect, rng)
        context_idx = block_to_indices(top, left, h, w, grid_w) - target_union

        if len(context_idx) >= min_context_patches:
            return sorted(context_idx), [sorted(b) for b in target_blocks]

    raise RuntimeError(
        f"could not sample a context block with >= {min_context_patches} "
        f"visible patches after {max_resample_attempts} attempts"
    )


def _render_grid(grid_h, grid_w, context_indices, target_blocks):
    """ASCII visualization: '.' unused, 'C' context, '1'-'4'/'+' target blocks
    (numbered by first block that claims a cell; '+' marks target/target overlap)."""
    cells = [["." for _ in range(grid_w)] for _ in range(grid_h)]
    for i, block in enumerate(target_blocks):
        label = str(i + 1) if i < 9 else "#"
        for idx in block:
            r, c = divmod(idx, grid_w)
            cells[r][c] = label if cells[r][c] == "." else "+"
    for idx in context_indices:
        r, c = divmod(idx, grid_w)
        assert cells[r][c] == ".", "context block must not overlap target blocks"
        cells[r][c] = "C"
    return "\n".join(" ".join(row) for row in cells)


def _sanity_check(grid_h, grid_w, context_indices, target_blocks):
    num_patches = grid_h * grid_w
    context_set = set(context_indices)

    assert all(0 <= i < num_patches for i in context_indices), "context index out of bounds"
    for b in target_blocks:
        assert all(0 <= i < num_patches for i in b), "target index out of bounds"

    for i, b in enumerate(target_blocks):
        overlap = context_set & set(b)
        assert not overlap, f"context overlaps target block {i} at patches {overlap}"

    assert len(context_indices) == len(context_set), "context indices not unique"
    for b in target_blocks:
        assert len(b) == len(set(b)), "target block indices not unique"

    assert len(context_indices) > 0, "context block is empty"
    for i, b in enumerate(target_blocks):
        assert len(b) > 0, f"target block {i} is empty"


if __name__ == "__main__":
    # Placeholder grid: patch size 4 on 32x32 CIFAR-10 -> 8x8 = 64 patches.
    # Easy to change once the actual encoder's patch size is decided.
    GRID_H, GRID_W = 8, 8

    rng = random.Random(0)
    context_indices, target_blocks = sample_masks(GRID_H, GRID_W, rng=rng)

    _sanity_check(GRID_H, GRID_W, context_indices, target_blocks)

    num_patches = GRID_H * GRID_W
    print(f"grid: {GRID_H}x{GRID_W} = {num_patches} patches\n")
    print(f"context block: {len(context_indices)} patches "
          f"({100 * len(context_indices) / num_patches:.1f}% of grid)")
    print(f"context indices: {context_indices}\n")

    for i, b in enumerate(target_blocks):
        print(f"target block {i + 1}: {len(b)} patches "
              f"({100 * len(b) / num_patches:.1f}% of grid)")
        print(f"  indices: {b}")

    print("\nall sanity checks passed\n")
    print(_render_grid(GRID_H, GRID_W, context_indices, target_blocks))