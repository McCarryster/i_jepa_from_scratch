import math
import random

# Taken from the paper
TARGET_BLOCK_SCALE = (0.15, 0.2)
CONTEXT_BLOCK_SCALE = (0.85, 1.0)
TARGET_BLOCK_ASPECT_RATIO = (0.75, 1.5)
CONTEXT_BLOCK_ASPECT_RATIO = (1.0, 1.0)


def sample_block(
        grid_height: int,
        grid_width: int,
        scale_range: tuple[float, float],
        aspect_ratio_range: tuple[float, float],
        rng: random.Random) -> tuple[int, int, int, int]:
    """
    Sample one rectangular block of patches from height*width grid.

    Args:
        grid_height: Number of rows in the grid.
        grid_width: Number of columns in the grid.
        scale_range: (low, high) fraction of total grid area the block should cover.
            - Pass (0.15, 0.2) (used for the I-JEPA target block).
            - Pass (0.85, 1.0) (used for the I-JEPA context block).
        aspect_ratio_range: (low, high) for block height/width ratio.
            - Pass (0.75, 1.5) for a random rectangle block (used for the I-JEPA target block).
            - Pass (1.0, 1.0) for a fixed-square block (used for the I-JEPA context block).
        rng: Random source to sample from.

    Returns:
        tuple(start_row, start_column, new_height, new_width) in patch coordinates.
    """

    # 1. Sample scale and aspect ratio
    scale = rng.uniform(*scale_range)
    aspect_ratio = rng.uniform(*aspect_ratio_range)

    # 2. Calculate total area
    num_patches = grid_height*grid_width # 2*4 = 8
    area = num_patches * scale # 8*0.5 = 4

    # 3. Calculate new height and width
    block_height = int(round(math.sqrt(area * aspect_ratio))) # sqrt(4*1.5)=2.4
    block_width = int(round(math.sqrt(area / aspect_ratio))) # sqrt(4/1.5)=1.6
    block_height = min(max(block_height, 1), grid_height)
    block_width = min(max(block_width, 1), grid_width)

    # 4. Randomly pick placement
    start_row = rng.randint(0, grid_height - block_height) # randint(0, 0)
    start_column = rng.randint(0, grid_width - block_width) # randint(0, 2)
    # 1 2 3 4    1 2 3
    # 5 6 7 8 -> 5 6 7

    return start_row, start_column, block_height, block_width


def block_to_indices(start_row: int,
                     start_column: int,
                     block_height: int,
                     block_width: int,
                     grid_width: int) -> set[int]:
    """
    Flattens a (start_row, start_column, block_height, block_width) rectangle into a set of row-major patch indices.

    Args:
        start_row: 1st row of the block.
        start_column: 1st column of the block.
        block_height: Block height, in patches.
        block_width: Block width, in patches.
        grid_width: Original grid width in patches.

    Returns:
        The set of flat patch indices covered by the block.
    """

    indices = set()
    for r in range(start_row, start_row+block_height): # 1, 1+2
        for c in range(start_column, start_column+block_width): # 1, 1+3
            index = r * grid_width + c
            indices.add(index)
    return indices

    # 1 line alternative
    # return {r * grid_w + c for r in range(top, top + h) for c in range(left, left + w)}


def sample_masks(grid_height: int,
                 grid_width: int,
                 rng: random.Random,
                 num_target_blocks: int = 4,
                 min_context_patches: int = 1,
                 max_resample_attempts: int = 20,) -> tuple[list[list[int]], list[int]]:

    """
    Samples one context block + num_target_blocks target blocks for one image.

    Mirrors the I-JEPA paper: target blocks are sampled independently (may overlap each other),
    then a single square context block is sampled and any patches overlapping ANY target block are removed from it.

    Note on batching: when this is wrapped in a data-loader collator, all images in a batch should share the same (h, w) for their context blocks
    and the same (h, w) for their target blocks (paper appendix), so the per-image index lists can be stacked.
    That batching concern is deliberately NOT handled here - this function only samples for a single image.

    Args:
        - grid_height: Number of rows in the grid.
        - grid_width: Number of columns in the grid.
        - rng: Random source to sample from.
        - num_target_blocks: Number of target blocks to sample.
        - min_context_patches: Minimum number of context patches required after removing target overlap; resamples everything if not met.
        - max_resample_attempts: Number of full resample attempts before giving up.

    Returns:
        - (target_block_indices, context_block_indices) tuple:
            - target_block_indices is a list of num_target_blocks sorted lists of patch indices, one per target block.
            - context_block_indices is a sorted list of patch indices visible to the context encoder.

    Raises:
        RuntimeError: couldn't find a valid context block after {max_resample_attempts} attempts.
    """

    for _ in range(max_resample_attempts):
        # 1. Sample target blocks and convert to indices
        target_block_indices = []
        for _ in range(num_target_blocks):
            start_row, start_column, block_height, block_width = sample_block(grid_height, grid_width, TARGET_BLOCK_SCALE, TARGET_BLOCK_ASPECT_RATIO, rng)
            target_block_indices.append(block_to_indices(start_row, start_column, block_height, block_width, grid_width))

        # 2. Merge target block indices for context block subtraction
        target_union = set()
        for tb_i in target_block_indices:
            target_union.update(tb_i)
        # 1 line alternative
        # target_union = set().union(*target_block_indices)

        # 3. Sample context block
        start_row, start_column, block_height, block_width = sample_block(grid_height, grid_width, CONTEXT_BLOCK_SCALE, CONTEXT_BLOCK_ASPECT_RATIO, rng)
        context_block_indices = block_to_indices(start_row, start_column, block_height, block_width, grid_width)
        context_block_indices = context_block_indices.difference(target_union)

        if len(context_block_indices) >= min_context_patches:
            return [sorted(block) for block in target_block_indices], sorted(context_block_indices)

    raise RuntimeError(f"couldn't find a valid context block after {max_resample_attempts} attempts")