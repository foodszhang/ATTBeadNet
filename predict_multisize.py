"""
Prediction script for K-class bead center detection on raw test images.

Outputs:
- pred_centers.csv: per-class center coordinates
- count_summary.csv: bead count per class
- overlays/: PNG with colored dots on grayscale
- heatmaps/: per-class probability heatmaps
"""

import argparse
import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import skimage as ski
import torch
from skimage import measure

from config.config import cfg
from model import build_unet3plus
from utils.mytransforms import min_max_normalization


def extract_peaks_per_class(
    prob_maps: np.ndarray,
    thresholds: List[float],
    min_distances: List[int]
) -> List[Dict]:
    """
    Extract peak centers per class from probability maps using peak_local_max.

    Args:
        prob_maps: np.ndarray [K, H, W] - probability maps for K classes
        thresholds: list[K] - threshold per class
        min_distances: list[K] - minimum distance between peaks per class

    Returns:
        List of dicts: [{"class_id": k, "x": int, "y": int, "score": float}, ...]
    """
    from skimage.feature import peak_local_max

    K, H, W = prob_maps.shape
    centers = []

    for k in range(K):
        coords = peak_local_max(
            prob_maps[k],
            min_distance=min_distances[k] if k < len(min_distances) else 3,
            threshold_abs=thresholds[k] if k < len(thresholds) else 0.5,
            exclude_border=False,
        )

        for y, x in coords:
            y = int(np.round(y))
            x = int(np.round(x))
            y = min(max(y, 0), H - 1)
            x = min(max(x, 0), W - 1)
            score = float(prob_maps[k, y, x])
            centers.append({
                "class_id": k,
                "x": x,
                "y": y,
                "score": score
            })

    return centers


def predict_tile(model: torch.nn.Module, tile: np.ndarray, device: str) -> np.ndarray:
    """
    Run prediction on a single tile.

    Args:
        model: trained UNet3+ model
        tile: tile [1, 1, H, W] as numpy (normalized to [-1, 1])
        device: "cuda" or "cpu"

    Returns:
        prob_maps [K, H, W] as numpy (probabilities in [0, 1])
    """
    tile_tensor = torch.from_numpy(tile).float().to(device)
    with torch.no_grad():
        outputs = model(tile_tensor)
        if isinstance(outputs, dict):
            outputs = outputs.get("final_pred", outputs.get("final"))
        # Handle training vs eval mode output shape
        if outputs.dim() == 4 and outputs.shape[0] == 1:
            outputs = outputs[0]  # [K, H, W]
        probs = torch.sigmoid(outputs)
    return probs.cpu().numpy()


def merge_tile_predictions(
    tiles_probs: List[np.ndarray],
    tile_coords: List[Tuple[int, int, int, int]],
    output_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Merge tile predictions by averaging overlapping regions.

    Args:
        tiles_probs: list of [K, tile_h, tile_w] probability arrays
        tile_coords: list of (y0, x0, h, w) coordinates for each tile
        output_shape: (H, W) output image shape

    Returns:
        merged [K, H, W] probability map
    """
    K = tiles_probs[0].shape[0]
    H, W = output_shape
    acc = np.zeros((K, H, W), dtype=np.float64)
    count = np.zeros((K, H, W), dtype=np.float64)

    for prob, (y0, x0, h, w) in zip(tiles_probs, tile_coords):
        acc[:, y0:y0+h, x0:x0+w] += prob
        count[:, y0:y0+h, x0:x0+w] += 1

    # Avoid division by zero
    count = np.maximum(count, 1)
    return acc / count


def make_overlay(
    image: np.ndarray,
    centers: List[Dict],
    class_names: List[str],
    class_colors: List[Tuple[int, int, int]] = None
) -> np.ndarray:
    """
    Draw colored dots on grayscale image for detected centers.

    Args:
        image: grayscale image [H, W] in [0, 1] range
        centers: list of {"class_id": k, "x": int, "y": int, "score": float}
        class_names: list of class names
        class_colors: list of (R, G, B) tuples, one per class

    Returns:
        RGB image [H, W, 3] as uint8
    """
    H, W = image.shape
    # Convert grayscale to RGB
    if image.dtype != np.uint8:
        gray = (image * 255).astype(np.uint8)
    else:
        gray = image.copy()
    overlay = ski.color.gray2rgb(gray)

    # Default colors: class 0 -> cyan, class 1 -> yellow, rest uses HSV
    if class_colors is None:
        class_colors = []
        for k in range(len(class_names)):
            if k == 0:
                class_colors.append((0, 255, 255))  # cyan
            elif k == 1:
                class_colors.append((255, 255, 0))  # yellow
            else:
                # HSV with hue based on class index
                h = int(360 * k / (len(class_names) + 1))
                rgb = ski.color.hsv2rgb([[[h/360, 1.0, 1.0]]])
                class_colors.append(tuple(int(x * 255) for x in rgb[0, 0]))

    # Draw dots at each center
    for center in centers:
        k = center["class_id"]
        x, y = center["x"], center["y"]
        r, g, b = class_colors[k % len(class_colors)]
        # Draw a small disk
        rr, cc = ski.draw.disk((y, x), radius=3, shape=(H, W))
        overlay[rr, cc] = [r, g, b]

    return overlay.astype(np.uint8)


def sliding_window(
    H: int, W: int, tile_size: int, stride: int
) -> List[Tuple[int, int, int, int]]:
    """
    Generate sliding window coordinates covering full image.

    Returns:
        List of (y0, x0, h, w) tuples
    """
    coords = []
    y = 0
    while y < H:
        h = tile_size if y + tile_size <= H else H - y
        x = 0
        while x < W:
            w = tile_size if x + tile_size <= W else W - x
            coords.append((y, x, h, w))
            x += stride
        y += stride
    return coords


def main(args):
    # Load config
    cfg.merge_from_file(args.cfg)
    cfg.freeze()

    num_classes = cfg.data.num_classes
    class_names = cfg.data.class_names if hasattr(cfg.data, 'class_names') else [str(i) for i in range(num_classes)]
    thresholds = cfg.postprocess.thresholds if hasattr(cfg.postprocess, 'thresholds') else [0.5] * num_classes
    min_distances = cfg.postprocess.min_distances if hasattr(cfg.postprocess, 'min_distances') else [3] * num_classes

    # Ensure thresholds and min_distances match num_classes
    if len(thresholds) < num_classes:
        thresholds = thresholds + [0.5] * (num_classes - len(thresholds))
    if len(min_distances) < num_classes:
        min_distances = min_distances + [3] * (num_classes - len(min_distances))

    # Build model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_unet3plus(
        num_classes,
        cfg.model.encoder,
        cfg.model.skip_ch,
        cfg.model.aux_losses,
        cfg.model.use_cgm,
        cfg.model.pretrained,
        cfg.model.dropout,
        am="CBAM",
    )

    # Load checkpoint
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model = model.to(device)

    # Create output directories
    os.makedirs(args.out_dir, exist_ok=True)
    overlays_dir = os.makedirs(os.path.join(args.out_dir, "overlays"), exist_ok=True)
    heatmaps_dir = os.makedirs(os.path.join(args.out_dir, "heatmaps"), exist_ok=True)

    # Prepare CSV files (overwrite for fresh run)
    pred_centers_file = os.path.join(args.out_dir, "pred_centers.csv")
    count_summary_file = os.path.join(args.out_dir, "count_summary.csv")
    # Start with empty files (headers will be written on first image)
    open(pred_centers_file, 'w').close()
    open(count_summary_file, 'w').close()

    # Process each image
    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.tif', '.tiff', '.jpg', '.png'))]
    image_files.sort()

    for imgname in image_files:
        img_path = os.path.join(args.input_dir, imgname)
        img = ski.io.imread(img_path)
        # Ensure grayscale single-channel
        if img.ndim == 3:
            img = ski.color.rgb2gray(img) if img.shape[-1] in [3, 4] else img[:, :, 0]
        img = img.astype(np.float32)

        H, W = img.shape
        pure_img = img.copy()

        # Normalize to [0, 1]
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img_norm = (img - img_min) / (img_max - img_min)
        else:
            img_norm = np.zeros_like(img)

        # Sliding window prediction
        tile_size = args.tile_size
        stride = args.stride
        tiles_probs = []
        tile_coords = []

        for y0, x0, h, w in sliding_window(H, W, tile_size, stride):
            tile = img_norm[y0:y0+h, x0:x0+w]
            # Pad to tile_size if needed
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                tile_padded = np.zeros((tile_size, tile_size), dtype=np.float32)
                tile_padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = tile_padded

            # Normalize to [-1, 1] like training
            tile = 2.0 * tile - 1.0
            # Add batch and channel dims: [1, 1, H, W]
            tile = np.expand_dims(np.expand_dims(tile, axis=0), axis=0)

            prob = predict_tile(model, tile, device)
            tiles_probs.append(prob)
            tile_coords.append((y0, x0, h, w))

        # Merge tile predictions
        merged_probs = merge_tile_predictions(tiles_probs, tile_coords, (H, W))  # [K, H, W]

        # Extract peaks per class
        centers = extract_peaks_per_class(merged_probs, thresholds, min_distances)

        # Count per class
        count_per_class = {k: 0 for k in range(num_classes)}
        for c in centers:
            count_per_class[c["class_id"]] += 1

        # Write pred_centers.csv (append)
        img_id = os.path.splitext(imgname)[0]
        with open(pred_centers_file, 'a') as f:
            for c in centers:
                f.write(f"{img_id},{c['class_id']},{class_names[c['class_id']]},{c['x']},{c['y']},{c['score']:.6f}\n")

        # Write count_summary.csv (append)
        total_count = sum(count_per_class.values())
        with open(count_summary_file, 'a') as f:
            row = [img_id]
            for k in range(num_classes):
                row.append(str(count_per_class[k]))
            row.append(str(total_count))
            f.write(",".join(row) + "\n")

        # Save overlay
        overlay = make_overlay(pure_img, centers, class_names)
        overlay_path = os.path.join(args.out_dir, "overlays", f"{img_id}.png")
        ski.io.imsave(overlay_path, overlay)

        # Save per-class heatmaps
        for k in range(num_classes):
            heatmap = merged_probs[k]
            # Normalize heatmap to [0, 1] for saving
            if heatmap.max() > heatmap.min():
                heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            else:
                heatmap_norm = heatmap
            heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)
            heatmap_path = os.path.join(args.out_dir, "heatmaps", f"{img_id}_class{k}.png")
            ski.io.imsave(heatmap_path, heatmap_uint8)

        print(f"Processed {imgname}: {len(centers)} centers detected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict bead centers on test images")
    parser.add_argument("--cfg", required=True, help="config file path")
    parser.add_argument("--ckpt", required=True, help="checkpoint path")
    parser.add_argument("--input-dir", required=True, help="directory with test images")
    parser.add_argument("--out-dir", required=True, help="output directory")
    parser.add_argument("--tile-size", type=int, default=64, help="tile size for sliding window")
    parser.add_argument("--stride", type=int, default=32, help="stride for sliding window")

    args = parser.parse_args()
    main(args)
