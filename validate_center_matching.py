"""Validate with center-based matching + Hungarian algorithm + threshold sweep."""
import argparse
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from config.config import cfg
from datasets import MultiSizeBeadTileDataset
from model import build_unet3plus
from utils.metrics import (
    extract_gt_centers_from_mask,
    extract_pred_centers_from_prob,
    hungarian_match,
)


def validate_with_center_matching(model, val_loader, cfg, device):
    """Validate using center-based matching with peak_local_max."""
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
        for batch in tqdm(val_loader, desc="Validating"):
            images = batch["image"].float().to(device)
            masks = batch["mask"].float().cpu().numpy()
            counts = batch["count"].float().cpu().numpy()

            out = model(images)
            logits = out["final_pred"] if isinstance(out, dict) else out
            probs = torch.sigmoid(logits).cpu().numpy()

            B = images.shape[0]
            for b in range(B):
                gt_centers = extract_gt_centers_from_mask(masks[b], mode="pixel")
                pred_centers = extract_pred_centers_from_prob(
                    probs[b], thresholds, min_distances, use_peak_local_max=True
                )

                matched, unmatched_pred, unmatched_gt = hungarian_match(
                    pred_centers, gt_centers, match_radius
                )

                for k in range(K):
                    pred_k = [p for p in pred_centers if p['class_id'] == k]
                    gt_k = [g for g in gt_centers if g['class_id'] == k]
                    matched_k = [m for m in matched if pred_centers[m[0]]['class_id'] == k]

                    tp[k] += len(matched_k)
                    fp[k] += len(pred_k) - len(matched_k)
                    fn[k] += len(gt_k) - len(matched_k)

                    pred_count = len(pred_k)
                    gt_count = int(counts[b, k])
                    count_err[k] += abs(pred_count - gt_count)
                    n_samples[k] += 1

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


def validate_with_threshold_sweep(model, val_loader, cfg, device, threshold_range):
    """Sweep thresholds and return best configuration."""
    K = cfg.data.num_classes
    min_distances = cfg.postprocess.min_distances
    match_radius = cfg.postprocess.match_radius
    class_names = cfg.data.class_names

    # Collect all probs and masks first
    all_probs = []
    all_masks = []
    all_counts = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Collecting predictions"):
            images = batch["image"].float().to(device)
            masks = batch["mask"].float().cpu().numpy()
            counts = batch["count"].float().cpu().numpy()

            out = model(images)
            logits = out["final_pred"] if isinstance(out, dict) else out
            probs = torch.sigmoid(logits).cpu().numpy()

            for b in range(images.shape[0]):
                all_probs.append(probs[b])
                all_masks.append(masks[b])
                all_counts.append(counts[b])

    results = []
    for t in threshold_range:
        test_thresholds = [t] * K

        tp = [0.0] * K
        fp = [0.0] * K
        fn = [0.0] * K
        count_err = [0.0] * K
        n_samples = [0] * K

        for prob, mask, count in zip(all_probs, all_masks, all_counts):
            gt_centers = extract_gt_centers_from_mask(mask, mode="pixel")
            pred_centers = extract_pred_centers_from_prob(
                prob, test_thresholds, min_distances, use_peak_local_max=True
            )

            matched, unmatched_pred, unmatched_gt = hungarian_match(
                pred_centers, gt_centers, match_radius
            )

            for k in range(K):
                pred_k = [p for p in pred_centers if p['class_id'] == k]
                gt_k = [g for g in gt_centers if g['class_id'] == k]
                matched_k = [m for m in matched if pred_centers[m[0]]['class_id'] == k]

                tp[k] += len(matched_k)
                fp[k] += len(pred_k) - len(matched_k)
                fn[k] += len(gt_k) - len(matched_k)

                pred_count = len(pred_k)
                gt_count = int(count[k])
                count_err[k] += abs(pred_count - gt_count)
                n_samples[k] += 1

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
            })
            f1s.append(f1)

        macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
        results.append({
            "threshold": t,
            "macro_f1": macro_f1,
            "per_class": rows,
        })
        print(f"  threshold={t:.2f}: macro_f1={macro_f1:.4f}", end="")
        for row in rows:
            print(f" [{row['class_name']}: F1={row['f1']:.3f}, P={row['precision']:.3f}, R={row['recall']:.3f}]", end="")
        print()

    return results


def match_radius_sweep(model, val_loader, cfg, device, match_radii):
    """Sweep match radius and return results."""
    K = cfg.data.num_classes
    thresholds = cfg.postprocess.thresholds
    min_distances = cfg.postprocess.min_distances
    class_names = cfg.data.class_names

    # Collect all data
    all_probs = []
    all_masks = []
    all_counts = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Collecting predictions"):
            images = batch["image"].float().to(device)
            masks = batch["mask"].float().cpu().numpy()
            counts = batch["count"].float().cpu().numpy()

            out = model(images)
            logits = out["final_pred"] if isinstance(out, dict) else out
            probs = torch.sigmoid(logits).cpu().numpy()

            for b in range(images.shape[0]):
                all_probs.append(probs[b])
                all_masks.append(masks[b])
                all_counts.append(counts[b])

    results = []
    for radius in match_radii:
        tp = [0.0] * K
        fp = [0.0] * K
        fn = [0.0] * K

        for prob, mask in zip(all_probs, all_masks):
            gt_centers = extract_gt_centers_from_mask(mask, mode="pixel")
            pred_centers = extract_pred_centers_from_prob(
                prob, thresholds, min_distances, use_peak_local_max=True
            )

            matched, unmatched_pred, unmatched_gt = hungarian_match(
                pred_centers, gt_centers, radius
            )

            for k in range(K):
                pred_k = [p for p in pred_centers if p['class_id'] == k]
                gt_k = [g for g in gt_centers if g['class_id'] == k]
                matched_k = [m for m in matched if pred_centers[m[0]]['class_id'] == k]

                tp[k] += len(matched_k)
                fp[k] += len(pred_k) - len(matched_k)
                fn[k] += len(gt_k) - len(matched_k)

        rows = []
        f1s = []
        for k in range(K):
            p = tp[k] / (tp[k] + fp[k]) if (tp[k] + fp[k]) > 0 else 0.0
            r = tp[k] / (tp[k] + fn[k]) if (tp[k] + fn[k]) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            rows.append({"class_id": k, "class_name": class_names[k], "precision": p, "recall": r, "f1": f1})
            f1s.append(f1)

        macro_f1 = sum(f1s) / len(f1s)
        results.append({"match_radius": radius, "macro_f1": macro_f1, "per_class": rows})
        print(f"  match_radius={radius}: macro_f1={macro_f1:.4f}", end="")
        for row in rows:
            print(f" [{row['class_name']}: F1={row['f1']:.3f}, R={row['recall']:.3f}]", end="")
        print()

    return results


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

    ckpt_path = args.ckpt if args.ckpt else os.path.join(
        cfg.train.log_dir, cfg.train.save_name, f"{cfg.train.save_name}_best.ckpt"
    )
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

    print(f"\n=== Center-based Validation ===")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Dataset: {cfg.data.train_root}")
    print(f"Current thresholds: {cfg.postprocess.thresholds}")
    print(f"Match radius: {cfg.postprocess.match_radius}")

    if args.sweep_thresholds:
        print("\n--- Threshold Sweep ---")
        threshold_range = [round(x * 0.05, 2) for x in range(3, 11)]  # 0.15 to 0.50
        sweep_results = validate_with_threshold_sweep(
            model, val_loader, cfg, device, threshold_range
        )
        # Find best
        best = max(sweep_results, key=lambda x: x["macro_f1"])
        print(f"\nBest threshold: {best['threshold']:.2f}, macro_f1: {best['macro_f1']:.4f}")

        # Save sweep results
        all_rows = []
        for r in sweep_results:
            for row in r["per_class"]:
                row_copy = {"threshold": r["threshold"]}
                row_copy.update(row)
                all_rows.append(row_copy)
        pd.DataFrame(all_rows).to_csv("threshold_sweep_results.csv", index=False)

    if args.sweep_radius:
        print("\n--- Match Radius Sweep ---")
        match_radii = [3, 4, 5, 6]
        radius_results = match_radius_sweep(model, val_loader, cfg, device, match_radii)

    # Standard validation with current config
    print("\n--- Validation with current config ---")
    metrics = validate_with_center_matching(model, val_loader, cfg, device)
    print(f"\nmacro_f1: {metrics['macro_f1']:.4f}")
    for row in metrics["per_class"]:
        print(
            f"  class {row['class_name']}: "
            f"P={row['precision']:.3f} R={row['recall']:.3f} "
            f"F1={row['f1']:.4f} MAE={row['mae_count']:.2f} "
            f"(TP={row['tp']}, FP={row['fp']}, FN={row['fn']})"
        )

    df = pd.DataFrame(metrics["per_class"])
    df.to_csv("val_metrics_center_matching.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate with center-based Hungarian matching")
    parser.add_argument("--cfg", default="config/multisize_bead.yaml", help="config file")
    parser.add_argument("--ckpt", default=None, help="checkpoint path (default: best.ckpt)")
    parser.add_argument("--sweep-thresholds", action="store_true", help="sweep threshold values")
    parser.add_argument("--sweep-radius", action="store_true", help="sweep match radius values")
    args = parser.parse_args()
    main(args)