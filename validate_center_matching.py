"""Validate with center-based matching + Hungarian algorithm."""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from skimage import measure
from tqdm import tqdm
from torch.utils.data import DataLoader

from config.config import cfg
from datasets import MultiSizeBeadTileDataset
from model import build_unet3plus


def extract_centers_from_mask(mask: np.ndarray) -> list:
    """Extract center coordinates from binary/heatmap mask."""
    # mask shape: [K, H, W] where values are 0.0 or 1.0
    centers = []
    K = mask.shape[0]
    for k in range(K):
        binary = mask[k] > 0.5
        labeled = measure.label(binary, connectivity=2)
        props = measure.regionprops(labeled)
        for prop in props:
            centroid = prop.centroid  # (row, col) = (y, x)
            centers.append({
                'class_id': k,
                'y': centroid[0],
                'x': centroid[1],
            })
    return centers


def extract_centers_from_prob(prob: np.ndarray, threshold: float, min_distance: int) -> list:
    """Extract centers from probability map using local maxima."""
    # prob shape: [K, H, W]
    centers = []
    K = prob.shape[0]
    for k in range(K):
        prob_k = prob[k]
        # Simple threshold + connected components for now
        binary = prob_k > threshold
        labeled = measure.label(binary, connectivity=2)
        props = measure.regionprops(labeled, intensity_image=prob_k)
        for prop in props:
            centroid = prop.centroid
            centers.append({
                'class_id': k,
                'y': centroid[0],
                'x': centroid[1],
                'score': prop.mean_intensity,
            })
    return centers


def compute_distance_matrix(pred_centers: list, gt_centers: list, match_radius: float) -> tuple:
    """Build cost matrix for Hungarian matching. Returns (cost_matrix, valid_pairs)."""
    if len(pred_centers) == 0 or len(gt_centers) == 0:
        return None, []

    # Filter by class first
    K = max(max(p['class_id'] for p in pred_centers), max(g['class_id'] for g in gt_centers)) + 1

    # Build cost matrix with infinity for class mismatch
    cost = np.full((len(pred_centers), len(gt_centers)), 1e9, dtype=np.float32)

    valid_pairs = []
    for i, pred in enumerate(pred_centers):
        for j, gt in enumerate(gt_centers):
            if pred['class_id'] == gt['class_id']:
                dist = np.sqrt((pred['x'] - gt['x'])**2 + (pred['y'] - gt['y'])**2)
                if dist <= match_radius:
                    cost[i, j] = dist
                    valid_pairs.append((i, j))

    return cost, valid_pairs


def hungarian_match(pred_centers: list, gt_centers: list, match_radius: float) -> tuple:
    """Match predictions to GT using Hungarian algorithm within match_radius."""
    cost, valid_pairs = compute_distance_matrix(pred_centers, gt_centers, match_radius)

    if cost is None or len(valid_pairs) == 0:
        return [], [], []  # matched, unmatched_pred, unmatched_gt

    # Run Hungarian on valid portion
    row_indices, col_indices = linear_sum_assignment(cost)

    matched = []
    unmatched_pred = list(range(len(pred_centers)))
    unmatched_gt = list(range(len(gt_centers)))

    for r, c in zip(row_indices, col_indices):
        if cost[r, c] <= match_radius:
            matched.append((r, c))
            if r in unmatched_pred:
                unmatched_pred.remove(r)
            if c in unmatched_gt:
                unmatched_gt.remove(c)

    return matched, unmatched_pred, unmatched_gt


def validate_with_center_matching(model, val_loader, cfg, device):
    """Validate using center-based matching with Hungarian algorithm."""
    K = cfg.data.num_classes
    thresholds = cfg.postprocess.thresholds
    min_distances = cfg.postprocess.min_distances
    match_radius = cfg.postprocess.match_radius
    class_names = cfg.data.class_names

    tp = [0.0] * K
    fp = [0.0] * K
    fn = [0.0] * K
    count_err = [0.0] * K
    n_samples = [0] * K

    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating (center matching)"):
            images = batch["image"].float().to(device)
            masks = batch["mask"].float().cpu().numpy()
            counts = batch["count"].float().cpu().numpy()

            out = model(images)
            logits = out["final_pred"] if isinstance(out, dict) else out
            probs = torch.sigmoid(logits).cpu().numpy()

            B = images.shape[0]
            for b in range(B):
                # Extract GT centers for this image
                gt_centers = extract_centers_from_mask(masks[b])  # list of {class_id, y, x}

                # Extract predicted centers for this image
                pred_centers = []
                for k in range(K):
                    thresh = thresholds[k] if k < len(thresholds) else 0.5
                    min_d = min_distances[k] if k < len(min_distances) else 3
                    binary = probs[b, k] > thresh
                    labeled = measure.label(binary, connectivity=2)
                    props = measure.regionprops(labeled, intensity_image=probs[b, k])
                    for prop in props:
                        pred_centers.append({
                            'class_id': k,
                            'y': prop.centroid[0],
                            'x': prop.centroid[1],
                            'score': prop.mean_intensity,
                        })

                # Hungarian matching
                matched, unmatched_pred, unmatched_gt = hungarian_match(
                    pred_centers, gt_centers, match_radius
                )

                # Per-class TP/FP/FN
                for k in range(K):
                    pred_k = [p for p in pred_centers if p['class_id'] == k]
                    gt_k = [g for g in gt_centers if g['class_id'] == k]
                    matched_k = [m for m in matched if pred_centers[m[0]]['class_id'] == k]

                    tp[k] += len(matched_k)
                    fp[k] += len(pred_k) - len(matched_k)
                    fn[k] += len(gt_k) - len(matched_k)

                    # Count MAE
                    pred_count = len(pred_k)
                    gt_count = int(counts[b, k])
                    count_err[k] += abs(pred_count - gt_count)
                    n_samples[k] += 1

    # Compute metrics
    rows = []
    f1s = []
    for k in range(K):
        p = tp[k] / (tp[k] + fp[k]) if (tp[k] + fp[k]) > 0 else 0.0
        r = tp[k] / (tp[k] + fn[k]) if (tp[k] + fn[k]) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        mae = count_err[k] / n_samples[k] if n_samples[k] > 0 else 0.0
        rows.append({
            "class_id": k,
            "class_name": class_names[k] if k < len(class_names) else str(k),
            "precision": p,
            "recall": r,
            "f1": f1,
            "mae_count": mae,
            "tp": int(tp[k]),
            "fp": int(fp[k]),
            "fn": int(fn[k]),
        })
        f1s.append(f1)

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return {"macro_f1": macro_f1, "per_class": rows}


def main(args):
    cfg.merge_from_file(args.cfg)
    cfg.freeze()

    device = cfg.train.device

    # Load model
    model = build_unet3plus(
        num_classes=cfg.data.num_classes,
        encoder=cfg.model.encoder,
        skip_ch=cfg.model.skip_ch,
        aux_losses=cfg.model.aux_losses,
        use_cgm=cfg.model.use_cgm,
        pretrained=cfg.model.pretrained,
        dropout=cfg.model.dropout,
    ).to(device)

    # Load checkpoint
    ckpt_path = args.ckpt
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Load validation dataset
    val_ds = MultiSizeBeadTileDataset(
        root_dir=cfg.data.train_root,
        split="internal_val",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    # Run validation
    metrics = validate_with_center_matching(model, val_loader, cfg, device)

    # Print results
    print(f"\n=== Center-based Validation Results ===")
    print(f"macro_f1: {metrics['macro_f1']:.4f}")
    for row in metrics["per_class"]:
        print(
            f"  class {row['class_name']}: "
            f"P={row['precision']:.3f} R={row['recall']:.3f} "
            f"F1={row['f1']:.4f} MAE={row['mae_count']:.2f} "
            f"(TP={row['tp']}, FP={row['fp']}, FN={row['fn']})"
        )

    # Save to CSV
    df = pd.DataFrame(metrics["per_class"])
    df.to_csv("val_metrics_center_matching.csv", index=False)
    print(f"\nSaved to val_metrics_center_matching.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate with center-based Hungarian matching")
    parser.add_argument("--cfg", default="config/multisize_bead.yaml", help="config file")
    parser.add_argument("--ckpt", required=True, help="checkpoint path")
    args = parser.parse_args()
    main(args)