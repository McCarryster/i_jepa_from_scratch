import math
import random

# TARGET_BLOCK_SCALE = (0.15, 0.2)
# CONTEXT_BLOCK_SCALE = (0.85, 1.0)
# TARGET_BLOCK_ASPECT_RATIO = (0.75, 1.5)
# CONTEXT_BLOCK_ASPECT_RATIO = (1.0, 1.0)

def sample_block(
        grid_height: int,
        grid_width: int,
        scale_range: tuple[float, float],
        aspect_ratio_range: tuple[float, float],
        rng: random.Random) -> tuple[int, int, int, int]:
    """
    Sample one rectangular block of patches from height*width grid.

    Args:
        patches_height: Number of rows in the grid.
        patches_width: Number of columns in the grid.
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
    new_height = int(round(math.sqrt(area * aspect_ratio))) # sqrt(4*1.5)=2.4
    new_width = int(round(math.sqrt(area / aspect_ratio))) # sqrt(4/1.5)=1.6
    new_height = min(max(new_height, 1), grid_height)
    new_width = min(max(new_width, 1), grid_width)

    # 4. Randomly pick placement
    start_row = rng.randint(0, grid_height - new_height) # randint(0, 0)
    start_column = rng.randint(0, grid_width - new_width) # randint(0, 2)
    # 1 2 3 4    1 2 3
    # 5 6 7 8 -> 5 6 7

    return start_row, start_column, new_height, new_width


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