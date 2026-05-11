"""D4 geometric transforms for multisize bead tile augmentation.

All transforms operate on masks of shape [K, H, W] with axes=(-2, -1).
"""

import numpy as np


def _identity(x, axes=None):
    return x


def _rot90(x, axes=None):
    return np.rot90(x, k=1, axes=axes)


def _rot180(x, axes=None):
    return np.rot90(x, k=2, axes=axes)


def _rot270(x, axes=None):
    return np.rot90(x, k=3, axes=axes)


def _flip_h(x, axes=None):
    return np.fliplr(x)


def _flip_v(x, axes=None):
    return np.flipud(x)


def _transpose(x, axes=None):
    # Swap H and W: for [K, H, W] use axes=(0, 2, 1), for [H, W] use axes=(1, 0)
    if x.ndim == 3:
        return np.transpose(x, axes=(0, 2, 1))
    else:
        return np.transpose(x, axes=(1, 0))


def _anti_transpose(x, axes=None):
    # Transpose then flip vertical
    return np.flipud(_transpose(x, axes=axes))


D4_TRANSFORMS = [
    ("identity", _identity),
    ("rot90", _rot90),
    ("rot180", _rot180),
    ("rot270", _rot270),
    ("flip_h", _flip_h),
    ("flip_v", _flip_v),
    ("transpose", _transpose),
    ("anti_transpose", _anti_transpose),
]

D4_NAMES = [name for name, _ in D4_TRANSFORMS]


def apply_d4(image, mask, aug_id, axes=(-2, -1)):
    """Apply D4 transform to image and mask.

    Args:
        image: [H, W] float32 image
        mask: [K, H, W] uint8 mask
        aug_id: Transform index (0-7)
        axes: Axes for rotation-based transforms

    Returns:
        (transformed_image, transformed_mask)
    """
    name, fn = D4_TRANSFORMS[aug_id]
    return fn(image, axes=axes), fn(mask, axes=axes)


def seeded_augment_ids(tile_id, n, global_seed=42, exclude_id=0):
    """Yield n augmentation IDs (0-7) deterministically.

    Args:
        tile_id: Unique tile identifier
        n: Number of augmentation IDs to yield
        global_seed: Global seed for reproducibility
        exclude_id: Always exclude this ID (default 0 = identity)

    Yields:
        int augmentation IDs in range [0, 7], excluding identity
    """
    # Available IDs are 1-7 (exclude 0)
    available_ids = [i for i in range(8) if i != exclude_id]

    for aug_idx in range(n):
        # Derive seed from inputs
        seed = hash((global_seed, tile_id, aug_idx)) % 2**31
        rng = np.random.RandomState(seed)

        if n > 7:
            # Cycle through with different seeds per aug_idx
            sample_id = available_ids[aug_idx % len(available_ids)]
        else:
            # Sample without replacement: shuffle once, then pick sequentially
            if aug_idx == 0:
                rng.shuffle(available_ids)
            sample_id = available_ids[aug_idx]

        yield sample_id
