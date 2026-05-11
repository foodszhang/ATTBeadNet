"""End-to-end CLI script for multisize bead tile preprocessing.

Pipeline:
1. CLI argument parsing
2. Image ID collection with mask validation
3. Image-level normalization (p1-p99.8 percentile clip → [0,1])
4. Sliding window tile extraction
5. Positive/negative tile filtering
6. D4 geometric augmentation (seeded, only for train_pool)
7. Tile writing (image .tif + mask .npy)
8. Debug overlay generation (first 16 tiles)
9. Manifest.csv generation
10. config.json generation
"""

import argparse
import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.augment_multisize import (
    D4_NAMES,
    apply_d4,
    seeded_augment_ids,
)
from utils.tile_utils import make_overlay, sliding_window


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_image(img, p_low=1, p_high=99.8):
    """Percentile clip and rescale image to [0, 1].

    Args:
        img: [H, W] float32 raw image
        p_low: Lower percentile for clipping (default 1)
        p_high: Upper percentile for clipping (default 99.8)

    Returns:
        [H, W] float32 normalized image in [0, 1]
    """
    lo = np.percentile(img, p_low)
    hi = np.percentile(img, p_high)
    img_clipped = np.clip(img, lo, hi)
    range_ = hi - lo
    if range_ > 0:
        img_norm = (img_clipped - lo) / range_
    else:
        img_norm = np.zeros_like(img_clipped, dtype=np.float32)
    return img_norm.astype(np.float32)


# ---------------------------------------------------------------------------
# Image ID collection
# ---------------------------------------------------------------------------

def collect_image_ids(raw_root, image_dir, class_dirs, mask_suffix, image_suffix, allow_missing=False):
    """Collect image IDs by scanning for images and verifying masks.

    Args:
        raw_root: Root directory containing raw data
        image_dir: Subdirectory containing images (e.g., "RF")
        class_dirs: List of mask subdirectory names
        mask_suffix: Suffix for mask files (e.g., "_Mask.tif")
        image_suffix: Suffix for image files (e.g., ".tif")
        allow_missing: If True, warn and skip images with missing masks

    Returns:
        Sorted list of valid image IDs (strings)
    """
    img_dir = os.path.join(raw_root, image_dir)
    if not os.path.isdir(img_dir):
        raise ValueError(f"Image directory not found: {img_dir}")

    # Find all image files
    image_files = {}
    for f in os.listdir(img_dir):
        if f.endswith(image_suffix):
            img_id = f[:-len(image_suffix)]
            image_files[img_id] = os.path.join(img_dir, f)

    if not image_files:
        raise ValueError(f"No images found in {img_dir} with suffix {image_suffix}")

    # Verify all masks exist for each image
    valid_ids = []
    for img_id in sorted(image_files.keys()):
        all_masks_found = True
        for class_dir in class_dirs:
            mask_path = os.path.join(raw_root, class_dir, f"{img_id}{mask_suffix}")
            if not os.path.isfile(mask_path):
                if allow_missing:
                    warnings.warn(f"Missing mask {mask_path} for image {img_id}, skipping")
                    all_masks_found = False
                    break
                else:
                    raise FileNotFoundError(f"Missing required mask: {mask_path}")
        if all_masks_found:
            valid_ids.append(img_id)

    return sorted(valid_ids)


# ---------------------------------------------------------------------------
# Train/val split
# ---------------------------------------------------------------------------

def split_by_image_id(image_ids, split_mode, val_ratio, global_seed):
    """Split image IDs into train and validation sets.

    Args:
        image_ids: List of image ID strings
        split_mode: "all_train", "train_val_by_image", or "fixed_test"
        val_ratio: Fraction of images to use for validation
        global_seed: Random seed for reproducibility

    Returns:
        (train_ids, val_ids) lists of image ID strings
    """
    rng = np.random.default_rng(global_seed)
    shuffled = list(rng.permutation(image_ids))

    if split_mode == "all_train":
        return shuffled, []
    elif split_mode == "train_val_by_image":
        n_val = max(1, int(len(shuffled) * val_ratio))
        val_ids = shuffled[:n_val]
        train_ids = shuffled[n_val:]
        return train_ids, val_ids
    elif split_mode == "fixed_test":
        return [], shuffled
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")


# ---------------------------------------------------------------------------
# Source dataset name derivation
# ---------------------------------------------------------------------------

def derive_source_dataset(out_dir):
    """Derive a short source dataset name from the output directory."""
    # Use the last component of the out_dir path
    name = os.path.basename(os.path.normpath(out_dir))
    # Replace non-alphanumeric with underscore
    name = "".join(c if c.isalnum() else "_" for c in name)
    return name


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess multisize bead tiles from raw images"
    )
    parser.add_argument("--raw-root", required=True,
                        help="Root directory containing raw data")
    parser.add_argument("--image-dir", default="RF",
                        help="Subdirectory containing images (default: RF)")
    parser.add_argument("--class-dirs", required=True, nargs="+",
                        help="Mask subdirectory names, e.g., 1.0_mask 2.8_mask")
    parser.add_argument("--class-names", nargs="+", default=None,
                        help="Class names (default: derived from class-dirs by stripping _mask suffix)")
    parser.add_argument("--mask-suffix", default="_Mask.tif",
                        help="Mask file suffix (default: _Mask.tif)")
    parser.add_argument("--image-suffix", default=".tif",
                        help="Image file suffix (default: .tif)")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for processed tiles")
    parser.add_argument("--tile-size", type=int, default=64,
                        help="Tile size (default: 64)")
    parser.add_argument("--stride", type=int, default=32,
                        help="Sliding window stride (default: 32)")
    parser.add_argument("--split", default="train_pool",
                        help="Split name (default: train_pool)")
    parser.add_argument("--split-mode", default="all_train",
                        choices=["all_train", "train_val_by_image", "fixed_test"],
                        help="Split mode (default: all_train)")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Validation ratio for train_val_by_image mode (default: 0.15)")
    parser.add_argument("--negative-ratio", type=float, default=1.0,
                        help="Fraction of negative tiles to keep (default: 1.0 = keep all)")
    parser.add_argument("--mask-center-mode", default="pixel",
                        help="Mask center mode (default: pixel)")
    parser.add_argument("--augment-offline", action="store_true",
                        help="Enable offline D4 augmentation")
    parser.add_argument("--augment-multiplier", type=int, default=4,
                        help="Number of augmentation variants per tile (default: 4)")
    parser.add_argument("--global-seed", type=int, default=42,
                        help="Global random seed (default: 42)")
    parser.add_argument("--allow-missing-mask", action="store_true",
                        help="Allow missing masks and skip those images")
    parser.add_argument("--save-overlay-limit", type=int, default=16,
                        help="Save debug overlay for first N tiles (default: 16)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Validate augment_offline for independent_test
    if args.split == "independent_test" and args.augment_offline:
        raise ValueError(
            "--augment-offline cannot be used with independent_test split"
        )

    # Derive class_names from class_dirs if not provided
    if args.class_names is None:
        args.class_names = [
            d.replace("_mask", "").replace("_Mask", "") for d in args.class_dirs
        ]

    num_classes = len(args.class_dirs)
    source_dataset = os.path.basename(os.path.normpath(args.raw_root))

    print(f"[preprocess_multisize_tiles] raw_root={args.raw_root}")
    print(f"[preprocess_multisize_tiles] out_dir={args.out_dir}")
    print(f"[preprocess_multisize_tiles] class_names={args.class_names}")
    print(f"[preprocess_multisize_tiles] tile_size={args.tile_size}, stride={args.stride}")
    print(f"[preprocess_multisize_tiles] split={args.split}, split_mode={args.split_mode}")
    print(f"[preprocess_multisize_tiles] augment_offline={args.augment_offline}, multiplier={args.augment_multiplier}")

    # -------------------------------------------------------------------------
    # Step 1: Collect image IDs
    # -------------------------------------------------------------------------
    print("\n[1/9] Collecting image IDs...")
    image_ids = collect_image_ids(
        args.raw_root, args.image_dir, args.class_dirs,
        args.mask_suffix, args.image_suffix,
        allow_missing=args.allow_missing_mask
    )
    print(f"  Found {len(image_ids)} valid image IDs")

    if not image_ids:
        raise ValueError("No valid images found after mask validation")

    # -------------------------------------------------------------------------
    # Step 2: Split by image ID
    # -------------------------------------------------------------------------
    print("\n[2/9] Splitting by image ID...")
    train_ids, val_ids = split_by_image_id(
        image_ids, args.split_mode, args.val_ratio, args.global_seed
    )

    # Assign splits to image IDs
    image_to_split = {}
    for tid in train_ids:
        image_to_split[tid] = "train_pool"
    for vid in val_ids:
        image_to_split[vid] = "internal_val"

    print(f"  train images: {len(train_ids)}, val images: {len(val_ids)}")

    # -------------------------------------------------------------------------
    # Step 3 & 4: Per-image processing - tile extraction
    # -------------------------------------------------------------------------
    print("\n[3/9] Extracting tiles from images...")

    all_tiles = []  # list of dicts with tile metadata

    for img_id in image_ids:
        # Load image
        img_path = os.path.join(args.raw_root, args.image_dir, f"{img_id}{args.image_suffix}")
        img = tifffile.imread(img_path)
        if img.ndim != 2:
            raise ValueError(f"Image {img_path} is not 2D: shape={img.shape}")
        img = img.astype(np.float32)

        # Normalize
        img_norm = normalize_image(img, p_low=1, p_high=99.8)

        # Load K masks
        mask_stack = []
        for class_dir in args.class_dirs:
            mask_path = os.path.join(args.raw_root, class_dir, f"{img_id}{args.mask_suffix}")
            m = tifffile.imread(mask_path)
            if m.ndim != 2:
                raise ValueError(f"Mask {mask_path} is not 2D: shape={m.shape}")
            # Convert to uint8 if needed
            if m.dtype != np.uint8:
                m = m.astype(np.uint8)
            mask_stack.append(m)

        mask_stack = np.stack(mask_stack, axis=0)  # [K, H, W]

        # Verify shapes match
        if mask_stack.shape[1:] != img_norm.shape:
            raise ValueError(
                f"Image/mask shape mismatch for {img_id}: "
                f"image={img_norm.shape}, mask={mask_stack.shape[1:]}"
            )

        # Sliding window
        for tile in sliding_window(img_norm, mask_stack, args.tile_size, args.stride):
            tile_split = image_to_split.get(img_id, "train")

            # Filter by split (only process tiles from images in our target split)
            # For train_pool: include all train images
            # For independent_test: include all images
            if args.split == "train_pool" and tile_split not in ("train_pool", "internal_val"):
                continue
            elif args.split == "independent_test" and tile_split != "internal_val":
                # independent_test only uses val images
                continue

            # Negative tile filtering
            if not tile["is_positive"]:
                rng_neg = np.random.default_rng(
                    hash((args.global_seed, img_id, tile["y0"], tile["x0"])) % (2**31)
                )
                if rng_neg.random() > args.negative_ratio:
                    continue

            tile["img_id"] = img_id
            tile["img_path"] = img_path
            tile["split"] = tile_split
            all_tiles.append(tile)

    print(f"  Extracted {len(all_tiles)} tiles before augmentation")

    # -------------------------------------------------------------------------
    # Step 5: Augmentation (train_pool only, only if --augment-offline)
    # -------------------------------------------------------------------------
    print("\n[4/9] Processing augmentation...")

    augmented_tiles = []

    for tile in all_tiles:
        tile_id_base = f"{source_dataset}_{tile['img_id']}_y{tile['y0']}_x{tile['x0']}"

        # Determine aug_ids to write
        if args.augment_offline and args.split == "train_pool":
            extra_aug_ids = list(seeded_augment_ids(
                tile_id_base,
                n=args.augment_multiplier,
                global_seed=args.global_seed,
                exclude_id=0
            ))
            aug_ids = [0] + extra_aug_ids
        else:
            aug_ids = [0]  # original only, no augmentation

        tile_image = tile["image"]
        tile_mask = tile["mask"]

        for aug_id in aug_ids:
            if aug_id == 0:
                aug_img = tile_image
                aug_mask = tile_mask
                geom_transform = "identity"
            else:
                aug_img, aug_mask = apply_d4(tile_image, tile_mask, aug_id, axes=(-2, -1))
                geom_transform = D4_NAMES[aug_id]

            if aug_id == 0:
                tile_id = tile_id_base
            else:
                tile_id = f"{tile_id_base}_aug{aug_id}"

            is_augmented = aug_id != 0
            source_tile_id = tile_id_base  # always points to original

            count_per_class = [
                int((aug_mask[k] > 0).sum()) for k in range(aug_mask.shape[0])
            ]
            count_total = sum(count_per_class)
            is_positive = count_total > 0

            augmented_tiles.append({
                "tile_id": tile_id,
                "source_dataset": source_dataset,
                "source_image_id": tile["img_id"],
                "source_image_path": tile["img_path"],
                "split": tile["split"],
                "image": aug_img,
                "mask": aug_mask,
                "y0": tile["y0"],
                "x0": tile["x0"],
                "tile_size": args.tile_size,
                "stride": args.stride,
                "class_names": args.class_names,
                "count_per_class": count_per_class,
                "count_total": count_total,
                "is_positive": is_positive,
                "augmentation_id": aug_id,
                "geom_transform": geom_transform,
                "source_tile_id": source_tile_id,
                "is_augmented": is_augmented,
            })

    print(f"  Total tiles after augmentation: {len(augmented_tiles)}")

    # -------------------------------------------------------------------------
    # Step 6: Write tiles to disk
    # -------------------------------------------------------------------------
    print("\n[5/9] Writing tiles to disk...")

    # Determine actual split for tile storage
    if args.split == "train_pool":
        storage_split = "train_pool"
    elif args.split == "independent_test":
        storage_split = "independent_test"
    else:
        storage_split = args.split

    images_out_dir = os.path.join(args.out_dir, storage_split, "images")
    masks_out_dir = os.path.join(args.out_dir, storage_split, "masks")
    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(masks_out_dir, exist_ok=True)

    debug_overlay_dir = os.path.join(args.out_dir, "debug", "tiles_overlay")
    os.makedirs(debug_overlay_dir, exist_ok=True)

    debug_counter = 0

    for tile in augmented_tiles:
        tile_id = tile["tile_id"]

        # Write image
        img_path = os.path.join(images_out_dir, f"{tile_id}.tif")
        tifffile.imwrite(img_path, tile["image"].astype(np.float32))

        # Write mask
        mask_path = os.path.join(masks_out_dir, f"{tile_id}_mask.npy")
        np.save(mask_path, tile["mask"].astype(np.uint8))

        # Update tile record with relative paths
        tile["image_path"] = os.path.join(storage_split, "images", f"{tile_id}.tif")
        tile["mask_path"] = os.path.join(storage_split, "masks", f"{tile_id}_mask.npy")

        # Debug overlay (first 16 tiles only)
        if debug_counter < args.save_overlay_limit:
            try:
                import cv2
                overlay = make_overlay(tile["image"], tile["mask"], tile["class_names"])
                overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                overlay_path = os.path.join(debug_overlay_dir, f"{tile_id}_overlay.png")
                cv2.imwrite(overlay_path, overlay_bgr)
            except Exception as e:
                warnings.warn(f"Failed to write debug overlay for {tile_id}: {e}")
            debug_counter += 1

    print(f"  Wrote {len(augmented_tiles)} tiles")

    # -------------------------------------------------------------------------
    # Step 7: Generate manifest.csv
    # -------------------------------------------------------------------------
    print("\n[6/9] Generating manifest.csv...")

    class_count_cols = [f"count_class_{i}" for i in range(num_classes)]

    manifest_rows = []
    for tile in augmented_tiles:
        row = {
            "tile_id": tile["tile_id"],
            "source_dataset": tile["source_dataset"],
            "source_image_id": tile["source_image_id"],
            "source_image_path": tile["source_image_path"],
            "split": tile["split"],
            "image_path": tile["image_path"],
            "mask_path": tile["mask_path"],
            "y0": tile["y0"],
            "x0": tile["x0"],
            "tile_size": tile["tile_size"],
            "stride": tile["stride"],
            "class_names": ",".join(tile["class_names"]),
            **{col: tile["count_per_class"][i] for i, col in enumerate(class_count_cols)},
            "count_total": tile["count_total"],
            "is_positive": tile["is_positive"],
            "augmentation_id": tile["augmentation_id"],
            "geom_transform": tile["geom_transform"],
            "source_tile_id": tile["source_tile_id"],
            "is_augmented": tile["is_augmented"],
        }
        manifest_rows.append(row)

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)
    print(f"  Wrote manifest with {len(manifest_df)} rows to {manifest_path}")

    # -------------------------------------------------------------------------
    # Step 8: Generate config.json
    # -------------------------------------------------------------------------
    print("\n[7/9] Generating config.json...")

    config = {
        "global_seed": args.global_seed,
        "data": {
            "raw_root": args.raw_root,
            "image_dir": args.image_dir,
            "class_dirs": args.class_dirs,
            "class_names": args.class_names,
            "mask_suffix": args.mask_suffix,
            "image_suffix": args.image_suffix,
            "processed_root": args.out_dir,
            "tile_size": args.tile_size,
            "stride": args.stride,
            "split_mode": args.split_mode,
            "val_ratio": args.val_ratio,
            "negative_ratio": args.negative_ratio,
            "mask_center_mode": args.mask_center_mode,
        },
        "augment": {
            "offline": args.augment_offline,
            "multiplier": args.augment_multiplier,
            "geom_transforms": D4_NAMES,
            "photometric": False,
        }
    }

    config_path = os.path.join(args.out_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Wrote config to {config_path}")

    print("\n[8/9] Pipeline complete!")
    print(f"  Output directory: {args.out_dir}")
    print(f"  Total tiles: {len(augmented_tiles)}")
    print(f"  Source dataset name: {source_dataset}")


if __name__ == "__main__":
    main()
