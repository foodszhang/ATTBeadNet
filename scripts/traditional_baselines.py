"""Traditional baseline experiments for bead center detection.

Methods:
- imagej_like: threshold/morphology/regionprops style detector.
- svm_patch: candidate peaks + patch SVM classifier.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import skimage.draw
import skimage.exposure
import skimage.feature
import skimage.filters
import skimage.io
import skimage.measure
import skimage.morphology
import tifffile
from scipy.optimize import linear_sum_assignment
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.preprocess_multisize_tiles import normalize_image
from utils.metrics import extract_gt_centers_from_mask, hungarian_match


CLASS_NAMES = ["1.0", "2.8"]
CLASS_DIRS = ["1.0_mask", "2.8_mask"]


def read_image_2d(path, is_mask=False):
    """Read a grayscale TIFF, tolerating RGB masks and mislabeled image files."""
    try:
        image = tifffile.imread(path)
    except Exception:
        from PIL import Image
        with Image.open(path) as pil_image:
            image = np.asarray(pil_image)
    if image.ndim == 3:
        image = image[..., :3].max(axis=-1) if is_mask else image[..., :3].mean(axis=-1)
    return image


def binarize_mask(mask):
    """Convert 0/1 or 0/255 masks to uint8 binary masks."""
    binary = mask >= 128 if mask.max(initial=0) > 1 else mask > 0
    if binary.mean() > 0.5:
        binary = ~binary
    return binary.astype(np.uint8)


def resolve_mask_path(raw_root, class_dir, image_id, mask_suffix="_Mask.tif"):
    """Resolve the common `_Mask.tif`/`_mask.tif` filename variants."""
    mask_dir = Path(raw_root) / class_dir
    for suffix in {mask_suffix, mask_suffix.replace("Mask", "mask"), mask_suffix.replace("mask", "Mask")}:
        candidate = mask_dir / f"{image_id}{suffix}"
        if candidate.is_file():
            return candidate
    return mask_dir / f"{image_id}{mask_suffix}"


def image_files(input_dir):
    """Return TIFF files in deterministic filename order."""
    suffixes = {".tif", ".tiff"}
    return sorted(
        (path for path in Path(input_dir).iterdir() if path.is_file() and path.suffix.lower() in suffixes),
        key=lambda path: path.name,
    )


def suppress_close_centers(centers, min_distance, class_aware=True):
    """Greedily retain the highest-scoring center within each local neighborhood."""
    if min_distance <= 0:
        return centers
    kept = []
    min_distance_sq = float(min_distance) ** 2
    for center in sorted(centers, key=lambda item: item.get("score", 0.0), reverse=True):
        if any(
            (not class_aware or center["class_id"] == accepted["class_id"])
            and (center["x"] - accepted["x"]) ** 2 + (center["y"] - accepted["y"]) ** 2 < min_distance_sq
            for accepted in kept
        ):
            continue
        kept.append(center)
    return sorted(kept, key=lambda item: (item["class_id"], item["y"], item["x"]))


def draw_overlay(image, centers, out_path, radius=7):
    """Save a two-color center overlay for qualitative comparison."""
    colors = [(0, 220, 255), (255, 210, 0)]
    background = (normalize_image(image.astype(np.float32)) * 255).astype(np.uint8)
    overlay = np.stack([background] * 3, axis=-1)
    for center in centers:
        y, x = int(round(center["y"])), int(round(center["x"]))
        color = colors[int(center["class_id"]) % len(colors)]
        for circle_radius in (radius, radius + 1):
            rr, cc = skimage.draw.circle_perimeter(y, x, radius=circle_radius, shape=image.shape)
            overlay[rr, cc] = color
    skimage.io.imsave(out_path, overlay, check_contrast=False)


def load_gt_centers(raw_root, image_id, class_dirs=CLASS_DIRS):
    masks = []
    for class_dir in class_dirs:
        mask_path = resolve_mask_path(raw_root, class_dir, image_id, "_Mask.tif")
        masks.append(binarize_mask(read_image_2d(mask_path, is_mask=True)))
    return extract_gt_centers_from_mask(np.stack(masks, axis=0), mode="pixel")


def split_image_ids(raw_root, processed_root, split):
    rf_ids = [p.stem for p in image_files(Path(raw_root) / "RF")]
    if split == "all" or not processed_root:
        return sorted(rf_ids, key=lambda x: int(x) if x.isdigit() else x)
    manifest = pd.read_csv(Path(processed_root) / "manifest.csv")
    ids = manifest[manifest["split"] == split]["source_image_id"].astype(str).unique().tolist()
    return sorted(ids, key=lambda x: int(x) if x.isdigit() else x)


def imagej_like_detect(image, min_distance=2):
    """ImageJ-style threshold + connected component candidate detector."""
    min_distance = max(1, int(round(min_distance)))
    img = normalize_image(image.astype(np.float32))
    enhanced = skimage.morphology.white_tophat(img, skimage.morphology.disk(4))
    enhanced = skimage.filters.gaussian(enhanced, sigma=0.7, preserve_range=True)
    if float(enhanced.max()) <= float(enhanced.min()):
        return []

    thr = skimage.filters.threshold_otsu(enhanced)
    binary = enhanced > max(thr * 0.75, np.percentile(enhanced, 97.0))
    labeled_small = skimage.measure.label(binary)
    binary = np.zeros_like(binary, dtype=bool)
    for prop in skimage.measure.regionprops(labeled_small):
        if prop.area >= 2:
            binary[labeled_small == prop.label] = True
    binary = skimage.morphology.closing(binary, skimage.morphology.disk(1))

    labeled = skimage.measure.label(binary)
    centers = []
    for prop in skimage.measure.regionprops(labeled, intensity_image=enhanced):
        if prop.area < 2 or prop.area > 80:
            continue
        perimeter = max(prop.perimeter, 1e-6)
        circularity = 4 * np.pi * prop.area / (perimeter * perimeter)
        if circularity < 0.25:
            continue
        y, x = prop.centroid_weighted if hasattr(prop, "centroid_weighted") else prop.weighted_centroid
        equivalent_diameter = (
            prop.equivalent_diameter_area
            if hasattr(prop, "equivalent_diameter_area")
            else prop.equivalent_diameter
        )
        max_intensity = prop.intensity_max if hasattr(prop, "intensity_max") else prop.max_intensity
        class_id = 1 if prop.area >= 14 or equivalent_diameter >= 4.2 else 0
        centers.append({
            "class_id": class_id,
            "y": float(y),
            "x": float(x),
            "score": float(max_intensity),
        })

    # Add LoG peaks as a Hough/blob-like supplement for isolated obvious dots.
    blobs = skimage.feature.blob_log(
        img,
        min_sigma=1.0,
        max_sigma=4.0,
        num_sigma=6,
        threshold=0.025,
        overlap=0.4,
    )
    for y, x, sigma in blobs:
        class_id = 1 if sigma >= 2.4 else 0
        centers.append({
            "class_id": class_id,
            "y": float(y),
            "x": float(x),
            "score": float(img[int(round(y)), int(round(x))]),
        })

    return suppress_close_centers(centers, min_distance=min_distance, class_aware=True)


def patch_features(image, y, x, radius=6):
    img = normalize_image(image.astype(np.float32))
    h, w = img.shape
    y = int(round(y))
    x = int(round(x))
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    patch = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
    patch[:y1 - y0, :x1 - x0] = img[y0:y1, x0:x1]
    cy = min(max(y, 0), h - 1)
    cx = min(max(x, 0), w - 1)
    feats = [
        patch.mean(), patch.std(), patch.max(), patch.min(),
        patch[radius, radius],
        patch.max() - patch.mean(),
        np.percentile(patch, 90), np.percentile(patch, 10),
    ]
    feats.extend(patch.flatten()[::2])
    return np.asarray(feats, dtype=np.float32)


def candidate_peaks(image, min_distance=2):
    min_distance = max(1, int(round(min_distance)))
    img = normalize_image(image.astype(np.float32))
    enhanced = skimage.morphology.white_tophat(img, skimage.morphology.disk(4))
    coords = skimage.feature.peak_local_max(
        enhanced,
        min_distance=min_distance,
        threshold_abs=max(0.01, float(np.percentile(enhanced, 92))),
        exclude_border=False,
    )
    return [{"y": float(y), "x": float(x), "score": float(enhanced[y, x])} for y, x in coords]


def train_svm(raw_root, train_ids, max_neg_per_image=800):
    xs = []
    ys = []
    rng = np.random.default_rng(42)
    for image_id in train_ids:
        image = read_image_2d(Path(raw_root) / "RF" / f"{image_id}.tif", is_mask=False)
        gt = load_gt_centers(raw_root, image_id)
        gt_xy = np.array([[g["y"], g["x"]] for g in gt], dtype=np.float32)

        for center in gt:
            xs.append(patch_features(image, center["y"], center["x"]))
            ys.append(int(center["class_id"]) + 1)

        candidates = candidate_peaks(image, min_distance=2)
        rng.shuffle(candidates)
        neg_count = 0
        for cand in candidates:
            if len(gt_xy):
                dist = np.sqrt(((gt_xy - np.array([cand["y"], cand["x"]])) ** 2).sum(axis=1)).min()
                if dist <= 6:
                    continue
            xs.append(patch_features(image, cand["y"], cand["x"]))
            ys.append(0)
            neg_count += 1
            if neg_count >= max_neg_per_image:
                break

    model = make_pipeline(
        StandardScaler(),
        SVC(C=2.0, gamma="scale", class_weight="balanced", probability=True),
    )
    model.fit(np.vstack(xs), np.asarray(ys, dtype=np.int64))
    return model


def svm_detect(image, model, min_distance=2, prob_threshold=0.30):
    centers = []
    candidates = candidate_peaks(image, min_distance=min_distance)
    if not candidates:
        return centers
    x = np.vstack([patch_features(image, c["y"], c["x"]) for c in candidates])
    pred = model.predict(x)
    prob = model.predict_proba(x)
    classes = list(model.classes_)
    for cand, label, probs in zip(candidates, pred, prob):
        if label == 0:
            continue
        label_prob = float(probs[classes.index(label)])
        if label_prob < prob_threshold:
            continue
        centers.append({
            "class_id": int(label) - 1,
            "y": cand["y"],
            "x": cand["x"],
            "score": label_prob,
        })
    return suppress_close_centers(centers, min_distance=min_distance, class_aware=True)


def match_centers_ignore_class(pred_centers, gt_centers, match_radius):
    if not pred_centers:
        return [], [], list(range(len(gt_centers)))
    if not gt_centers:
        return [], list(range(len(pred_centers))), []
    cost = np.full((len(pred_centers), len(gt_centers)), 1e9, dtype=np.float32)
    for i, pred in enumerate(pred_centers):
        for j, gt in enumerate(gt_centers):
            cost[i, j] = np.hypot(pred["x"] - gt["x"], pred["y"] - gt["y"])
    rows, cols = linear_sum_assignment(cost)
    matched = []
    unmatched_pred = set(range(len(pred_centers)))
    unmatched_gt = set(range(len(gt_centers)))
    for row, col in zip(rows, cols):
        if cost[row, col] <= match_radius:
            matched.append((row, col))
            unmatched_pred.discard(row)
            unmatched_gt.discard(col)
    return matched, sorted(unmatched_pred), sorted(unmatched_gt)


def evaluate_method(raw_root, image_ids, out_dir, method_name, detector, match_radius=3, save_overlays=True):
    out_dir = Path(out_dir) / method_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_overlays:
        (out_dir / "overlays").mkdir(exist_ok=True)
        (out_dir / "gt_overlays").mkdir(exist_ok=True)

    metric_rows = []
    center_rows = []
    confusion_rows = []
    for image_id in image_ids:
        image = read_image_2d(Path(raw_root) / "RF" / f"{image_id}.tif", is_mask=False)
        pred_centers = detector(image)
        gt_centers = load_gt_centers(raw_root, image_id)
        matched, _, _ = hungarian_match(pred_centers, gt_centers, match_radius)
        confusion_matched, confusion_unmatched_pred, confusion_unmatched_gt = match_centers_ignore_class(
            pred_centers, gt_centers, match_radius
        )

        if save_overlays:
            draw_overlay(image, pred_centers, out_dir / "overlays" / f"{image_id}_pred.png")
            draw_overlay(image, gt_centers, out_dir / "gt_overlays" / f"{image_id}_gt.png")

        for center in pred_centers:
            center_rows.append({
                "image_id": image_id,
                "class_id": center["class_id"],
                "class_name": CLASS_NAMES[center["class_id"]],
                "x": center["x"],
                "y": center["y"],
                "score": center.get("score", 0.0),
            })

        for class_id, class_name in enumerate(CLASS_NAMES):
            pred_k = [p for p in pred_centers if p["class_id"] == class_id]
            gt_k = [g for g in gt_centers if g["class_id"] == class_id]
            matched_k = [m for m in matched if pred_centers[m[0]]["class_id"] == class_id]
            tp = len(matched_k)
            fp = len(pred_k) - tp
            fn = len(gt_k) - tp
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            metric_rows.append({
                "image_id": image_id,
                "class_id": class_id,
                "class_name": class_name,
                "gt": len(gt_k),
                "pred": len(pred_k),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mae_count": abs(len(pred_k) - len(gt_k)),
            })

        for pred_idx, gt_idx in confusion_matched:
            pred = pred_centers[pred_idx]
            gt = gt_centers[gt_idx]
            gt_class = int(gt["class_id"])
            pred_class = int(pred["class_id"])
            confusion_rows.append({
                "image_id": image_id,
                "gt_class_name": CLASS_NAMES[gt_class],
                "pred_class_name": CLASS_NAMES[pred_class],
                "status": "correct" if gt_class == pred_class else "class_confusion",
                "count": 1,
            })
        for gt_idx in confusion_unmatched_gt:
            gt_class = int(gt_centers[gt_idx]["class_id"])
            confusion_rows.append({
                "image_id": image_id,
                "gt_class_name": CLASS_NAMES[gt_class],
                "pred_class_name": "miss",
                "status": "miss",
                "count": 1,
            })
        for pred_idx in confusion_unmatched_pred:
            pred_class = int(pred_centers[pred_idx]["class_id"])
            confusion_rows.append({
                "image_id": image_id,
                "gt_class_name": "false_positive",
                "pred_class_name": CLASS_NAMES[pred_class],
                "status": "false_positive",
                "count": 1,
            })

    metrics = pd.DataFrame(metric_rows)
    centers = pd.DataFrame(center_rows)
    confusion = pd.DataFrame(confusion_rows)
    metrics.to_csv(out_dir / "metrics_per_image.csv", index=False)
    centers.to_csv(out_dir / "pred_centers.csv", index=False)
    confusion.to_csv(out_dir / "confusion_matches.csv", index=False)
    summary = (
        metrics.groupby(["class_id", "class_name"], as_index=False)
        .agg(gt=("gt", "sum"), pred=("pred", "sum"), tp=("tp", "sum"), fp=("fp", "sum"), fn=("fn", "sum"),
             mae_count=("mae_count", "mean"))
    )
    summary["precision"] = summary["tp"] / (summary["tp"] + summary["fp"]).replace(0, np.nan)
    summary["recall"] = summary["tp"] / (summary["tp"] + summary["fn"]).replace(0, np.nan)
    summary["f1"] = 2 * summary["precision"] * summary["recall"] / (summary["precision"] + summary["recall"])
    summary = summary.fillna(0.0)
    summary.to_csv(out_dir / "metrics_summary.csv", index=False)
    matrix = pd.pivot_table(
        confusion,
        values="count",
        index=["gt_class_name"],
        columns=["pred_class_name"],
        aggfunc="sum",
        fill_value=0,
    )
    matrix.to_csv(out_dir / "confusion_matrix.csv")
    print(f"\n[{method_name}]")
    print(summary.to_string(index=False))
    print("macro_f1", float(summary["f1"].mean()))
    return summary


def main(args):
    train_ids = split_image_ids(args.raw_root, args.processed_root, "train_pool")
    eval_ids = split_image_ids(args.raw_root, args.processed_root, args.eval_split)
    print("train ids:", train_ids)
    print("eval ids:", eval_ids)

    summaries = []
    if "imagej_like" in args.methods:
        summary = evaluate_method(
            args.raw_root,
            eval_ids,
            args.out_dir,
            "imagej_like",
            lambda image: imagej_like_detect(image, min_distance=args.min_distance),
            args.match_radius,
            args.save_overlays,
        )
        summary.insert(0, "method", "imagej_like")
        summaries.append(summary)

    if "svm_patch" in args.methods:
        svm = train_svm(args.raw_root, train_ids, max_neg_per_image=args.max_neg_per_image)
        summary = evaluate_method(
            args.raw_root,
            eval_ids,
            args.out_dir,
            "svm_patch",
            lambda image: svm_detect(
                image,
                svm,
                min_distance=args.min_distance,
                prob_threshold=args.svm_threshold,
            ),
            args.match_radius,
            args.save_overlays,
        )
        summary.insert(0, "method", "svm_patch")
        summaries.append(summary)

    if summaries:
        all_summary = pd.concat(summaries, ignore_index=True)
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        all_summary.to_csv(Path(args.out_dir) / "traditional_baselines_summary.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run traditional bead detection baselines.")
    parser.add_argument("--raw-root", default="datasets/20260512")
    parser.add_argument("--processed-root", default="datasets/processed/20260512_tiles_96")
    parser.add_argument("--out-dir", default="runs/traditional_baselines")
    parser.add_argument("--methods", nargs="+", default=["imagej_like", "svm_patch"],
                        choices=["imagej_like", "svm_patch"])
    parser.add_argument("--eval-split", default="internal_val", choices=["internal_val", "train_pool", "all"])
    parser.add_argument("--match-radius", type=float, default=3.0)
    parser.add_argument("--min-distance", type=float, default=2.0)
    parser.add_argument("--svm-threshold", type=float, default=0.30)
    parser.add_argument("--max-neg-per-image", type=int, default=800)
    parser.add_argument("--save-overlays", action="store_true")
    main(parser.parse_args())
