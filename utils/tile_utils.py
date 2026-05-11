"""Sliding window tile extraction and overlay visualization utilities."""

import numpy as np


def sliding_window(image, mask, tile_size=64, stride=32):
    """
    Generate tiles using a sliding window over the image and mask.

    Args:
        image: [H, W] float32 array
        mask: [K, H, W] uint8 array (K classes)
        tile_size: size of each tile (default 64)
        stride: stride between tiles (default 32)

    Yields:
        dicts with keys: image, mask, y0, x0, tile_size, count_per_class,
        count_total, is_positive
    """
    H, W = image.shape
    for y0 in range(0, H - tile_size + 1, stride):
        for x0 in range(0, W - tile_size + 1, stride):
            # Skip tiles that would extend beyond image boundaries
            if y0 + tile_size > H or x0 + tile_size > W:
                continue

            image_tile = image[y0:y0 + tile_size, x0:x0 + tile_size]
            mask_tile = mask[:, y0:y0 + tile_size, x0:x0 + tile_size]

            count_per_class = [
                int((mask_tile[k] > 0).sum()) for k in range(mask.shape[0])
            ]
            count_total = sum(count_per_class)
            is_positive = count_total > 0

            yield {
                "image": image_tile,
                "mask": mask_tile,
                "y0": y0,
                "x0": x0,
                "tile_size": tile_size,
                "count_per_class": count_per_class,
                "count_total": count_total,
                "is_positive": is_positive,
            }


def make_overlay(image, mask, class_names=None, tile_size=64):
    """
    Create an RGB overlay visualization of the mask on the image.

    Args:
        image: [H, W] float in [0, 1]
        mask: [K, H, W] uint8 center mask
        class_names: list of class names (optional)
        tile_size: size of tiles (default 64, unused but kept for API)

    Returns:
        [H, W, 3] uint8 RGB overlay
    """
    import colorsys

    H, W = image.shape
    K = mask.shape[0]

    # Grayscale background from image
    image_01 = np.clip(image, 0.0, 1.0)
    gray = (image_01 * 255).astype(np.uint8)
    overlay = np.stack([gray, gray, gray], axis=-1)  # [H, W, 3]

    # Build per-class colors using HSV color wheel
    for k in range(K):
        if K == 2 and k == 0:
            h = 0.5  # cyan (180 deg)
        elif K == 2 and k == 1:
            h = 1/6  # yellow (60 deg)
        else:
            h = k / K
        s = 1.0
        v = 1.0
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        r_i = int(round(r * 255))
        g_i = int(round(g * 255))
        b_i = int(round(b * 255))

        # Find center of each connected component for this class
        mask_k = mask[k]
        positions = np.argwhere(mask_k > 0)
        for pos in positions:
            y, x = pos
            if 0 <= y < H and 0 <= x < W:
                overlay[y, x] = [r_i, g_i, b_i]

    return overlay
